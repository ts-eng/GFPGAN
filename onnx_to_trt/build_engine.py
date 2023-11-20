import os
import sys
import logging
import argparse

import numpy as np
import tensorrt as trt
from cuda import cudart

sys.path.insert(1, os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
import common

from image_batcher import ImageBatcher

logging.basicConfig(level=logging.INFO)
logging.getLogger("EngineBuilder").setLevel(logging.INFO)
log = logging.getLogger("EngineBuilder")


class MyLogger(trt.ILogger):
    def __init__(self, min_severity, log_path=None):
        trt.ILogger.__init__(self)
        self.min_severity = min_severity
        self.log_path = log_path

    def log(self, severity, msg):
        if severity >= self.min_severity:
            if self.log_path is not None:
                with open(self.log_path, 'a') as file:
                    file.write(f'{msg}\n')
            else:
                log.info(f'{msg}')


class EngineCalibrator(trt.IInt8EntropyCalibrator2):
    """
    Implements the INT8 Entropy Calibrator 2.
    """

    def __init__(self, cache_file):
        """
        :param cache_file: The location of the cache file.
        """
        super().__init__()
        self.cache_file = cache_file
        self.image_batcher = None
        self.batch_allocation = None
        self.batch_generator = None

    def set_image_batcher(self, image_batcher: ImageBatcher):
        """
        Define the image batcher to use, if any. If using only the cache file, an image batcher doesn't need
        to be defined.
        :param image_batcher: The ImageBatcher object
        """
        self.image_batcher = image_batcher
        size = int(np.dtype(self.image_batcher.dtype).itemsize * np.prod(self.image_batcher.shape))
        self.batch_allocation = common.cuda_call(cudart.cudaMalloc(size))
        self.batch_generator = self.image_batcher.get_batch()

    def get_batch_size(self):
        """
        Overrides from trt.IInt8EntropyCalibrator2.
        Get the batch size to use for calibration.
        :return: Batch size.
        """
        if self.image_batcher:
            return self.image_batcher.batch_size
        return 1

    def get_batch(self, names):
        """
        Overrides from trt.IInt8EntropyCalibrator2.
        Get the next batch to use for calibration, as a list of device memory pointers.
        :param names: The names of the inputs, if useful to define the order of inputs.
        :return: A list of int-casted memory pointers.
        """
        if not self.image_batcher:
            return None
        try:
            batch, _ = next(self.batch_generator)
            log.info("Calibrating image {} / {}".format(self.image_batcher.image_index, self.image_batcher.num_images))
            common.memcpy_host_to_device(self.batch_allocation, np.ascontiguousarray(batch))
            return [int(self.batch_allocation)]
        except StopIteration:
            log.info("Finished calibration batches")
            return None

    def read_calibration_cache(self):
        """
        Overrides from trt.IInt8EntropyCalibrator2.
        Read the calibration cache file stored on disk, if it exists.
        :return: The contents of the cache file, if any.
        """
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                log.info("Using calibration cache file: {}".format(self.cache_file))
                return f.read()

    def write_calibration_cache(self, cache):
        """
        Overrides from trt.IInt8EntropyCalibrator2.
        Store the calibration cache to a file on disk.
        :param cache: The contents of the calibration cache to store.
        """
        with open(self.cache_file, "wb") as f:
            log.info("Writing calibration cache data to: {}".format(self.cache_file))
            f.write(cache)


class EngineBuilder:
    """
    Parses an ONNX graph and builds a TensorRT engine from it.
    """

    def __init__(self, verbose=False, log_path=None):
        """
        :param verbose: If enabled, a higher verbosity level will be set on the TensorRT logger.
        """
        self.trt_logger = MyLogger(trt.Logger.VERBOSE, log_path)
        if verbose:
            self.trt_logger.min_severity = trt.Logger.Severity.VERBOSE

        trt.init_libnvinfer_plugins(self.trt_logger, namespace="")

        self.builder = trt.Builder(self.trt_logger)
        self.config = self.builder.create_builder_config()
        self.config.max_workspace_size = 8 * (2 ** 30)  # 8 GB

        self.batch_size = None
        self.network = None
        self.parser = None

    def create_network(self, onnx_path):
        """
        Parse the ONNX graph and create the corresponding TensorRT network definition.
        :param onnx_path: The path to the ONNX graph to load.
        """
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

        self.network = self.builder.create_network(network_flags)
        self.parser = trt.OnnxParser(self.network, self.trt_logger)

        onnx_path = os.path.realpath(onnx_path)
        with open(onnx_path, "rb") as f:
            if not self.parser.parse(f.read()):
                log.error("Failed to load ONNX file: {}".format(onnx_path))
                for error in range(self.parser.num_errors):
                    log.error(self.parser.get_error(error))
                sys.exit(1)

        inputs = [self.network.get_input(i) for i in range(self.network.num_inputs)]
        outputs = [self.network.get_output(i) for i in range(self.network.num_outputs)]

        log.info("Network Description")
        for input in inputs:
            self.batch_size = input.shape[0]
            log.info("Input '{}' with shape {} and dtype {}".format(input.name, input.shape, input.dtype))
        for output in outputs:
            log.info("Output '{}' with shape {} and dtype {}".format(output.name, output.shape, output.dtype))
        assert self.batch_size > 0
        self.builder.max_batch_size = self.batch_size
    
    def print_layer_info(self, filename):
        with open(filename, 'w') as file:
            for i in range(self.network.num_layers):
                layer = self.network.get_layer(i)
                file.write(f'{layer.type}\t{layer.name}\n')
    
    def set_mixed_precision(self):
        """
        Experimental precision mode.
        Enable mixed-precision mode.
        """
        self.config.set_flag(trt.BuilderFlag.STRICT_TYPES)

        for i in range(self.network.num_layers):
            layer = self.network.get_layer(i)
            if any(["modulated_conv/Pow" in layer.name,
                    "modulated_conv/ReduceSum" in layer.name,
                    "modulated_conv/Add" in layer.name,
                    "modulated_conv/Sqrt" in layer.name,
                    "modulated_conv/Div" in layer.name]):
                self.network.get_layer(i).precision = trt.DataType.FLOAT
                log.info("Mixed-Precision Layer {} set to FLOAT STRICT data type".format(layer.name))

    def create_engine(
        self,
        engine_path,
        precision,
        calib_input=None,
        calib_cache=None,
        calib_num_images=25000,
        calib_batch_size=8,
        calib_preprocessor=None,
    ):
        """
        Build the TensorRT engine and serialize it to disk.
        :param engine_path: The path where to serialize the engine to.
        :param precision: The datatype to use for the engine, either 'fp32', 'fp16' or 'int8'.
        :param calib_input: The path to a directory holding the calibration images.
        :param calib_cache: The path where to write the calibration cache to, or if it already exists, load it from.
        :param calib_num_images: The maximum number of images to use for calibration.
        :param calib_batch_size: The batch size to use for the calibration process.
        :param calib_preprocessor: The ImageBatcher preprocessor algorithm to use.
        """
        engine_path = os.path.realpath(engine_path)
        engine_dir = os.path.dirname(engine_path)
        os.makedirs(engine_dir, exist_ok=True)
        log.info("Building {} Engine in {}".format(precision, engine_path))

        inputs = [self.network.get_input(i) for i in range(self.network.num_inputs)]

        if precision == "fp16":
            if not self.builder.platform_has_fast_fp16:
                log.warning("FP16 is not supported natively on this platform/device")
            else:
                self.config.set_flag(trt.BuilderFlag.FP16)
        elif precision == "int8":
            if not self.builder.platform_has_fast_int8:
                log.warning("INT8 is not supported natively on this platform/device")
            else:
                self.config.set_flag(trt.BuilderFlag.INT8)
                self.config.int8_calibrator = EngineCalibrator(calib_cache)
                if not os.path.exists(calib_cache):
                    calib_shape = [calib_batch_size] + list(inputs[0].shape[1:])
                    calib_dtype = trt.nptype(inputs[0].dtype)
                    self.config.int8_calibrator.set_image_batcher(
                        ImageBatcher(
                            calib_input,
                            calib_shape,
                            calib_dtype,
                            max_num_images=calib_num_images,
                            exact_batches=True,
                            preprocessor=calib_preprocessor,
                        )
                    )

        with self.builder.build_engine(self.network, self.config) as engine, open(engine_path, "wb") as f:
            log.info("Serializing engine to file: {:}".format(engine_path))
            f.write(engine.serialize())


def main(args):
    log_path = args.onnx.replace('.onnx', '.log')
    builder = EngineBuilder(args.verbose, log_path)
    builder.create_network(args.onnx)
    builder.print_layer_info(args.onnx.replace('onnx', 'layers'))
    builder.set_mixed_precision()
    builder.create_engine(
        args.engine,
        args.precision,
        args.calib_input,
        args.calib_cache,
        args.calib_num_images,
        args.calib_batch_size,
        args.calib_preprocessor,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--onnx", help="The input ONNX model file to load")
    parser.add_argument("-e", "--engine", help="The output path for the TRT engine")
    parser.add_argument(
        "-p",
        "--precision",
        default="fp16",
        choices=["fp32", "fp16", "int8"],
        help="The precision mode to build in, either 'fp32', 'fp16' or 'int8', default: 'fp16'",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable more verbose log output")
    parser.add_argument("--calib_input", help="The directory holding images to use for calibration")
    parser.add_argument(
        "--calib_cache",
        default="./calibration.cache",
        help="The file path for INT8 calibration cache to use, default: ./calibration.cache",
    )
    parser.add_argument(
        "--calib_num_images",
        default=25000,
        type=int,
        help="The maximum number of images to use for calibration, default: 25000",
    )
    parser.add_argument(
        "--calib_batch_size", default=8, type=int, help="The batch size for the calibration process, default: 1"
    )
    parser.add_argument(
        "--calib_preprocessor",
        default="gfpganv1.4",
        choices=["gfpganv1.4"],
        help="Set the calibration image preprocessor to use, default: gfpganv1.4",
    )
    args = parser.parse_args()
    if not all([args.onnx, args.engine]):
        parser.print_help()
        log.error("These arguments are required: --onnx and --engine")
        sys.exit(1)
    if args.precision == "int8" and not any([args.calib_input, args.calib_cache]):
        parser.print_help()
        log.error("When building in int8 precision, either --calib_input or --calib_cache are required")
        sys.exit(1)
    main(args)