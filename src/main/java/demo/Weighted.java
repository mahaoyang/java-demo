package demo;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;

import java.util.Arrays;
import java.util.logging.Logger;

public class Weighted {
    public static void main(String[] args) {
        Module mod = Module.load("torch_weighted.pt");
        Tensor data =
                Tensor.fromBlob(
                        new double[]{1., 2., 3., 2., 6., 4., 4., 5., 6., 7., 8., 9.}, //data 物料个数*特征个数n
                        new long[]{4, 3} //shape 物料个数*特征个数n
                );
        Tensor weight =
                Tensor.fromBlob(
                        new double[]{1.0, 1.0, 1.0}, //data 特征个数n
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
