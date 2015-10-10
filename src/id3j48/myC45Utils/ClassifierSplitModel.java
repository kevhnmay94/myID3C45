package id3j48.myC45Utils;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

public abstract class ClassifierSplitModel {
    public Distribution classDistribution;
    public int numOfSubsets;

    public abstract int getSubsetIndex(Instance instance);
    public abstract double [] getWeights(Instance instance);

    public Instances[] split(Instances dataSet)
    {
        Instances [] subDataset = new Instances[numOfSubsets];
        double weights[];
        double newWeight;
        Instance instance;
        int subset;

        for(int i=0; i<numOfSubsets; i++)
        {
            subDataset[i] = new Instances(dataSet, dataSet.numInstances());
        }

        for(int i=0; i<dataSet.numInstances(); i++)
        {
            instance = dataSet.instance(i);
            weights = getWeights(instance);
            subset = getSubsetIndex(instance);
            if(subset > -1)
            {
                subDataset[subset].add(instance);
            }
            else
            {
                for(int j=0; j<numOfSubsets; j++)
                {
                    if(Utils.gr(weights[j],0))
                    {
                        newWeight = weights[j] * instance.weight();
                        subDataset[j].add(instance);
                        subDataset[j].lastInstance().setWeight(newWeight);
                    }
                }
            }
        }

        for(int j=0; j<numOfSubsets; j++)
        {
            subDataset[j].compactify();
        }

        return subDataset;
    }

    public final String toString(int index, Instances data) throws Exception {

        StringBuffer text;

        text = new StringBuffer();
        text.append(((Instances)data).classAttribute().value(classDistribution.maxClass(index)));
        text.append(" \n("+Utils.roundDouble(classDistribution.weightPerSubDataset[index],2));
        if (Utils.gr(classDistribution.numIncorrect(index),0))
            text.append("/"+Utils.roundDouble(classDistribution.numIncorrect(index),2));
        text.append(")");

        return text.toString();
    }

    public abstract String leftSide(Instances data);
    public abstract String rightSide(int index,Instances data);

    public double classProbability(int classIndex, Instance instance, int subsetIndex) {
        if(subsetIndex > -1)
        {
            return classDistribution.probability(classIndex, subsetIndex);
        }
        else
        {
            double [] weights = getWeights(instance);
            if(weights == null)
            {
                return classDistribution.probability(classIndex);
            }
            else
            {
                double probability = 0;
                for(int i=0; i<weights.length; i++)
                {
                    probability += weights[i] * classDistribution.probability(classIndex, i);
                }
                return probability;
            }
        }
    }
}
