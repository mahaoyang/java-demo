package demo;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;

import java.util.Arrays;
import java.util.logging.Logger;

public class App {
    public static void main(String[] args) {
        Module mod = Module.load("torch_weighted.pt1");
//        Tensor data =
//                Tensor.fromBlob(
//                        new float[]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, //data
//                        new long[]{6,} //shape
//                );
        Tensor data =
                Tensor.fromBlob(
                        new double[]{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, //data
                        new long[]{6,} //shape
                );
//        Tensor data =
//                Tensor.fromBlob(
//                        new long[]{6,}, //shape
//                        new double[]{1.0, 2.0, 3.0, 4.0, 5.0, 6.0} //data
//                );
        IValue result = mod.forward(IValue.from(data), IValue.from(3.0));
        Tensor output = result.toTensor();
        Logger logger = Logger.getLogger("test");
        logger.info("shape: " + Arrays.toString(output.shape()));
        logger.info("data: " + Arrays.toString(output.getDataAsDoubleArray()));
//        logger.info("data: " + Arrays.toString(output.getDataAsFloatArray()));

        // Workaround for https://github.com/facebookincubator/fbjni/issues/25
        System.exit(0);
    }
}
