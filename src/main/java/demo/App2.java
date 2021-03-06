package demo;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;

import java.util.Arrays;
import java.util.logging.Logger;

public class App2 {
    public static void main(String[] args) {
        Module mod = Module.load("torch_weighted.pt");
//        Tensor data =
//                Tensor.fromBlob(
//                        new float[]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, //data
//                        new long[]{6,} //shape
//                );
        Tensor data =
                Tensor.fromBlob(
                        new double[]{1., 2., 3., 2., 6., 4., 4., 5., 6.}, //data
                        new long[]{3, 3} //shape
                );
        Tensor weight =
                Tensor.fromBlob(
                        new double[]{1.0, 1.0, 1.0}, //data
                        new long[]{1, 3} //shape
                );
//        Tensor data =
//                Tensor.fromBlob(
//                        new long[]{6,}, //shape
//                        new double[]{1.0, 2.0, 3.0, 4.0, 5.0, 6.0} //data
//                );
        IValue result = mod.forward(IValue.from(data), IValue.from(weight));
        Tensor output = result.toTensor();
        Logger logger = Logger.getLogger("test");
        logger.info("shape: " + Arrays.toString(output.shape()));
        logger.info("data: " + Arrays.toString(output.getDataAsDoubleArray()));
//        logger.info("data: " + Arrays.toString(output.getDataAsFloatArray()));

        // Workaround for https://github.com/facebookincubator/fbjni/issues/25
        System.exit(0);
    }
}
