package id3j48.myC45Utils;

import weka.core.Instance;
import weka.core.Instances;

public class NoSplitTree extends ClassifierSplitModel{

    public NoSplitTree(Distribution distribution)
    {
        classDistribution = new Distribution(distribution);
        numOfSubsets = 1;
    }

    @Override
    public int getSubsetIndex(Instance instance) {
        return 0;
    }

    @Override
    public double[] getWeights(Instance instance) {
        return null;
    }

    @Override
    public final String leftSide(Instances instances){

        return "";
    }

    @Override
    public final String rightSide(int index, Instances instances){

        return "";
    }
}
