package id3j48;

import static id3j48.WekaAccess.readArff;
import static id3j48.WekaAccess.readCsv;
import java.util.Enumeration;
import java.util.Scanner;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.Id3;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.NoSupportForMissingValuesException;
import weka.core.Utils;

public class myID3 extends Classifier {
    private myID3[] children;
    private Attribute splitAttribute;
    private double classValue;
    private double[] classDistribution;
    private Attribute classAttribute;

    @Override
    public void buildClassifier(Instances instances) throws Exception {
        getCapabilities().testWithFail(instances);

        Instances data = new Instances(instances);
        data.deleteWithMissingClass();

        createTree(data);
    }
    
    private void createTree(Instances data) {
        if (data.numInstances() == 0) {
            splitAttribute = null;
            classValue = Instance.missingValue();
            classDistribution = new double[data.numClasses()];
        }
        else{
            double infoGains [] = new double[data.numAttributes()];
            Enumeration attributeEnumeration = data.enumerateAttributes();
            while(attributeEnumeration.hasMoreElements())
            {
                Attribute attribute = (Attribute) attributeEnumeration.nextElement();
                double infoGain = computeInfoGain(data, attribute);
                infoGains[attribute.index()] = infoGain;
            }

            splitAttribute = data.attribute(Utils.maxIndex(infoGains));

            if(Utils.eq(infoGains[splitAttribute.index()],0))
            {
                splitAttribute = null;
                classDistribution = new double[data.numClasses()];
                Enumeration instancesEnumeration = data.enumerateInstances();
                while(instancesEnumeration.hasMoreElements())
                {
                    Instance instance = (Instance) instancesEnumeration.nextElement();
                    classDistribution[((int) instance.classValue())]++;
                }
                Utils.normalize(classDistribution);
                classValue = Utils.maxIndex(classDistribution);
                classAttribute = data.classAttribute();
            }
            else
            {
                Instances[] subDataSet = splitData(data, splitAttribute);
                children = new myID3[splitAttribute.numValues()];
                for(int i=0; i<splitAttribute.numValues(); i++)
                {
                    children[i] = new myID3();
                    children[i].createTree(subDataSet[i]);
                }
            }
        }
        
        
    }

    private void createTree(Instances dataSet, double maxClassValue)
    {
        classAttribute = dataSet.classAttribute();
        classDistribution = new double[dataSet.numClasses()];

        if(dataSet.numInstances() == 0)
        {
            splitAttribute = null;
            classValue = maxClassValue;
        }
        else
        {
            double infoGains [] = new double[dataSet.numAttributes()];
            Enumeration attributeEnumeration = dataSet.enumerateAttributes();
            while(attributeEnumeration.hasMoreElements())
            {
                Attribute attribute = (Attribute) attributeEnumeration.nextElement();
                double infoGain = computeInfoGain(dataSet, attribute);
                infoGains[attribute.index()] = infoGain;
            }

            splitAttribute = dataSet.attribute(Utils.maxIndex(infoGains));

            if(Utils.eq(infoGains[splitAttribute.index()],0))
            {
                splitAttribute = null;
                Enumeration instancesEnumeration = dataSet.enumerateInstances();
                while(instancesEnumeration.hasMoreElements())
                {
                    Instance instance = (Instance) instancesEnumeration.nextElement();
                    classDistribution[((int) instance.classValue())]++;
                }
                Utils.normalize(classDistribution);
                classValue = Utils.maxIndex(classDistribution);

            }
            else
            {
                Instances[] subDataSet = splitData(dataSet, splitAttribute);
                children = new myID3[splitAttribute.numValues()];
                for(int i=0; i<splitAttribute.numValues(); i++)
                {
                    children[i] = new myID3();
                    children[i].createTree(subDataSet[i], maxClassValue);
                }
            }
        }
    }

    private double computeInfoGain(Instances data, Attribute attribute)
    {

        double infoGain = computeEntropy(data);
        Instances[] splitData = splitData(data, attribute);
        for (int j = 0; j < attribute.numValues(); j++) {
          if (splitData[j].numInstances() > 0) {
            infoGain -= ((double) splitData[j].numInstances() /
                         (double) data.numInstances()) *
            computeEntropy(splitData[j]);
          }
        }
        return infoGain;
    }

    private double computeEntropy(Instances data)
    {
        double [] classDistributions = new double[data.numClasses()];
        Enumeration instancesEnumeration = data.enumerateInstances();
        while(instancesEnumeration.hasMoreElements()){
            Instance ddata = (Instance) instancesEnumeration.nextElement();
            classDistributions[((int) ddata.classValue())]++;
        }

        double entropy = 0;
        for (int j = 0; j < data.numClasses(); j++) {
          if (classDistributions[j] > 0) {
            entropy -= classDistributions[j] * Utils.log2(classDistributions[j]);
          }
        }

        entropy /= (double) data.numInstances();
        return entropy + log2(data.numInstances());
    }

    private double log2(double a) {
        if(a == 0)
            return 0;
        else
            return Math.log(a) / Math.log(2);
    }

    private Instances[] splitData(Instances dataSet, Attribute attribute)
    {
        Instances [] subDataSet = new Instances [attribute.numValues()];
        for(int i=0; i<attribute.numValues(); i++)
        {
            subDataSet[i] = new Instances(dataSet, dataSet.numInstances());
        }

        Enumeration instancesEnumeration = dataSet.enumerateInstances();
        while(instancesEnumeration.hasMoreElements())
        {
            Instance instance = (Instance) instancesEnumeration.nextElement();
            subDataSet[((int) instance.value(attribute))].add(instance);
        }

        /* Return the empty array from each data set */
        for(int i=0; i<attribute.numValues(); i++)
        {
            subDataSet[i].compactify();
        }
        return subDataSet;
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        if (instance.hasMissingValue()) {
            throw new NoSupportForMissingValuesException("Missing values");
        }
        if (splitAttribute == null) {
            return classValue;
        } else {
            return children[(int) instance.value(splitAttribute)].
                    classifyInstance(instance);
        }
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        if (instance.hasMissingValue()) {
            throw new NoSupportForMissingValuesException("Missing values");
        }
        if (splitAttribute == null) {
            return classDistribution;
        } else {
            return children[(int) instance.value(splitAttribute)].
                    distributionForInstance(instance);
        }
    }

    @Override
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();
        result.enable(Capabilities.Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capabilities.Capability.NOMINAL_CLASS);
        result.enable(Capabilities.Capability.MISSING_CLASS_VALUES);

        result.setMinimumNumberInstances(0);

        return result;
    }

    @Override
    public String getRevision() {
        return super.getRevision();
    }

    private String toString(int level) {

        StringBuffer text = new StringBuffer();

        if (splitAttribute == null) {
            if (Instance.isMissingValue(classValue)) {
                text.append(": null");
            } else {
                text.append(": " + classAttribute.value((int) classValue));
            }
        } else {
            for (int j = 0; j < splitAttribute.numValues(); j++) {
                text.append("\n");
                for (int i = 0; i < level; i++) {
                    text.append("|  ");
                }
                text.append(splitAttribute.name() + " = " + splitAttribute.value(j));
                text.append(children[j].toString(level + 1));
            }
        }
        return text.toString();
    }

    @Override
    public String toString() {

        if ((classDistribution == null) && (children == null)) {
            return "-- Empty ID3 --";
        }
        return "ID3\n" + toString(0);
    }

    public static void main (String [] args) {
        try
        {
            WekaAccess.initializePath();
            Instances data = null;
            Scanner cin = new Scanner(System.in);
            System.out.println("Enter filename below");
            String filename = cin.nextLine();
            System.out.println("Loading "+filename+"...");
            String extension = "";
            String name = "";
            int i = filename.lastIndexOf('.');
            if (i > 0) {
                extension = filename.substring(i+1);
                name = filename.substring(0,i);
            }   if(extension.equalsIgnoreCase("arff")){
                try {
                    data = readArff(filename);
                } catch (Exception ex) {
                    Logger.getLogger(WekaAccess.class.getName()).log(Level.SEVERE, null, ex);
                }
            }
            else if(extension.equalsIgnoreCase("csv")){
                try {
                    data = readCsv(filename);
                } catch (Exception ex) {
                    Logger.getLogger(WekaAccess.class.getName()).log(Level.SEVERE, null, ex);
                }
            }
            else{
                System.out.println("Invalid extension");
                System.exit(0);
            }
            
            Classifier ID3classifier = new Id3();
            ID3classifier.buildClassifier(data);
            System.out.println(ID3classifier.toString());
            
            Evaluation ID3eval = WekaAccess.tenFoldCrossValidation(data, new Id3());
            System.out.println("\n-- Id3 result --");
            System.out.println(ID3eval.toSummaryString());

            Classifier myID3classifier = new myID3();
            myID3classifier.buildClassifier(data);
            System.out.println(myID3classifier.toString());
            
            Evaluation myID3eval = WekaAccess.tenFoldCrossValidation(data, new myID3());
            System.out.println("\n-- myID3 result --");
            System.out.println(myID3eval.toSummaryString());
            
            
        }
        catch (Exception e)
        {
            e.printStackTrace();
        }
    }

}
