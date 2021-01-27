package demo;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;

import java.util.Arrays;
import java.util.logging.Logger;

public class App {
    public static void main(String[] args) {
        Module mod = Module.load("torch_weighted.pt");
        Tensor data =
                Tensor.fromBlob(
                        new double[]{1., 2., 3., 2., 6., 4., 4., 5., 6., 7., 8., 9.}, //data
                        new long[]{3, 4} //shape
                );
        Tensor weight =
                Tensor.fromBlob(
                        new double[]{1.0, 1.0, 1.0}, //data
                        new long[]{1, 3} //shape
                );
        IValue result = mod.forward(IValue.from(data), IValue.from(weight));
        Tensor output = result.toTensor();
        Logger logger = Logger.getLogger("test");
        logger.info("shape: " + Arrays.toString(output.shape()));
        logger.info("data: " + Arrays.toString(output.getDataAsDoubleArray()));

        // Workaround for https://github.com/facebookincubator/fbjni/issues/25
        System.exit(0);
    }
}
