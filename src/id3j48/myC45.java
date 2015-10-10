package id3j48;

import static id3j48.WekaAccess.readArff;
import static id3j48.WekaAccess.readCsv;
import id3j48.myC45Utils.SplitTree;
import id3j48.myC45Utils.NoSplitTree;
import id3j48.myC45Utils.Distribution;
import id3j48.myC45Utils.ClassifierSplitModel;
import java.util.Enumeration;
import java.util.Scanner;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Statistics;
import weka.core.Utils;

public class myC45 extends Classifier{

    private myC45 [] children;
    private boolean is_leaf;
    private boolean is_empty;
    private Instances dataSet;
    private double minimalInstances = 2;
    private Distribution testSetDistribution;
    private double confidence = 0.1;
    private ClassifierSplitModel nodeType;
    Instances [] subDataset;

    @Override
    public void buildClassifier(Instances instances) throws Exception {
        getCapabilities().testWithFail(instances);

        Instances data = new Instances(instances);
        data.deleteWithMissingClass();

        build(data);
        collapse();
        prune();
    }

    private void build(Instances data){
        dataSet = data;
        is_leaf = false;
        is_empty = false;
        testSetDistribution = null;

        nodeType = processNode();
        if(nodeType.numOfSubsets > 1) {
            subDataset = nodeType.split(dataSet);
            children = new myC45[nodeType.numOfSubsets];
            for(int i=0; i<nodeType.numOfSubsets; i++) {
                children[i] = buildTree(subDataset[i]);
            }
        }
        else {
            is_leaf = true;
            if(Utils.eq(dataSet.sumOfWeights(), 0)) {
                is_empty = true;
            }
        }
    }
    
    public void collapse() {
        double subtreeError;
        double treeError;

        if (!is_leaf) {
            subtreeError = getTrainingError();
            treeError = nodeType.classDistribution.numIncorrect();
            if(subtreeError >= treeError-0.25) {
                children = null;
                is_leaf = true;
                nodeType = new NoSplitTree(nodeType.classDistribution);
            }
            else {
                for (int i=0; i<children.length; i++){
                    children[i].collapse();
                }
            }
        }
        
    }

    private void prune() {
        int largestBranchIndex;
        double largestBranchError;
        double leafError;
        double treeError;
        myC45 largestBranch;

        if(!is_leaf)
        {
            for(int i=0; i<children.length; i++){
                children[i].prune();
            }

            largestBranchIndex = Utils.maxIndex(nodeType.classDistribution.weightPerSubDataset);
            largestBranchError = children[largestBranchIndex].getBranchError(dataSet);
            leafError = getDistributionError(nodeType.classDistribution);
            treeError = getEstimatedError();

            if(Utils.smOrEq(leafError, treeError+0.1) && Utils.smOrEq(leafError, largestBranchError+0.1)){
                children = null;
                is_leaf = true;
                nodeType = new NoSplitTree(nodeType.classDistribution);
            }
            else if(Utils.smOrEq(largestBranchError, treeError + 0.1)){
                largestBranch = children[largestBranchIndex];
                children = largestBranch.children;
                nodeType = largestBranch.nodeType;
                is_leaf = largestBranch.is_leaf;
                prune();
            }
        }
    }

    private void createNewDistribution(Instances dataSet) {
        Instances [] subDataset;
        this.dataSet = dataSet;
        nodeType.classDistribution = new Distribution(dataSet);
        if(!is_leaf){
            subDataset = nodeType.split(dataSet);
            for(int i=0; i<children.length; i++){
                children[i].createNewDistribution(subDataset[i]);
            }
        }
        else{
            if(!Utils.eq(0, dataSet.sumOfWeights())){
                is_empty = false;
            }
            else{
                is_empty = true;
            }
        }
    }
    
    private double calculateError(double totalWeight, double numIncorect, double confidenceLevel) {
        if(numIncorect < 1){
            double base = totalWeight * (1 - Math.pow(confidenceLevel, 1 / totalWeight));
            if (numIncorect == 0) {
                return base;
            }
            else {
                return base + numIncorect * (calculateError(totalWeight, 1, confidenceLevel) - base);
            }
        }
        else {
            if (Utils.grOrEq(numIncorect + 0.5, totalWeight)) {
                return Math.max(totalWeight - numIncorect, 0);
            }
            else {
                double z = Statistics.normalInverse(1 - confidenceLevel);

                double f = (numIncorect + 0.5) / totalWeight;
                double r = (f + (z*z) / (2 * totalWeight) + z * Math.sqrt((f / totalWeight) - (f * f / totalWeight) + (z * z / (4 * totalWeight * totalWeight)))) / (1 + (z * z) / totalWeight);

                return (r * totalWeight) - numIncorect;
            }
        }
    }

    private double getEstimatedError() {
        double error = 0;

        if(is_leaf){
            return getDistributionError(nodeType.classDistribution);
        }
        else {
            for (int i=0; i<children.length; i++){
                error = error + children[i].getEstimatedError();
            }
            return error;
        }
    }

    private double getBranchError(Instances dataSet) {
        Instances [] subDataset;
        double error = 0;

        if(is_leaf) {
            return getDistributionError(new Distribution(dataSet));
        }
        else {
            Distribution tempClassDistribution = nodeType.classDistribution;
            nodeType.classDistribution = new Distribution(dataSet);
            subDataset = nodeType.split(dataSet);
            nodeType.classDistribution = tempClassDistribution;
            for(int i=0; i<children.length; i++) {
                error = error + children[i].getBranchError(subDataset[i]);
                return error;
            }
        }
        return 0;
    }

    private double getDistributionError(Distribution classDistribution) {
        if(Utils.eq(0, classDistribution.getTotalWeight())) {
            return 0;
        }
        else {
            return classDistribution.numIncorrect() + calculateError(classDistribution.getTotalWeight(), classDistribution.numIncorrect(), confidence);
        }
    }

    private myC45 buildTree(Instances subDataset) {
        myC45 newTree = new myC45();
        newTree.build(subDataset);
        return newTree;
    }

    private ClassifierSplitModel processNode()
    {
        double minGainRatio;
        SplitTree[] splitables;
        SplitTree bestSplitable = null;
        NoSplitTree notSplitable = null;
        double averageInfoGain = 0;
        int usefulSplitables = 0;
        Distribution classDistribution;
        double totalWeight;

        try{
            classDistribution = new Distribution(dataSet);
            notSplitable = new NoSplitTree(classDistribution);

            if(Utils.sm(dataSet.numInstances(), 2 * minimalInstances) ||
               Utils.eq(classDistribution.weightTotal, classDistribution.weightPerClass[Utils.maxIndex(classDistribution.weightPerClass)]))
            {
                return notSplitable;
            }

            splitables = new SplitTree[dataSet.numAttributes()];

            Enumeration attributeEnumeration = dataSet.enumerateAttributes();
            while(attributeEnumeration.hasMoreElements()){
                Attribute attribute = (Attribute) attributeEnumeration.nextElement();
                splitables[attribute.index()] = new SplitTree(attribute, minimalInstances, dataSet.sumOfWeights());
                splitables[attribute.index()].buildClassifier(dataSet);
                if(splitables[attribute.index()].validateNode()){
                    if(dataSet != null) {
                        averageInfoGain = averageInfoGain +  splitables[attribute.index()].infoGain;
                        usefulSplitables++;
                    }
                }
            }

            if (usefulSplitables == 0)
            {
                return notSplitable;
            }
            averageInfoGain = averageInfoGain/(double)usefulSplitables;

            minGainRatio = 0;
            attributeEnumeration = dataSet.enumerateAttributes();
            while(attributeEnumeration.hasMoreElements())
            {
                Attribute attribute = (Attribute) attributeEnumeration.nextElement();
                if(splitables[attribute.index()].validateNode())
                {
                    if(splitables[attribute.index()].infoGain >= (averageInfoGain - 0.001) &&
                       Utils.gr(splitables[attribute.index()].gainRatio, minGainRatio))
                    {
                        bestSplitable = splitables[attribute.index()];
                        minGainRatio = bestSplitable.gainRatio;
                    }
                }
            }

            if (Utils.eq(minGainRatio,0))
            {
                return notSplitable;
            }

            bestSplitable.addInstanceWithMissingvalue();

            if(dataSet != null)
            {
                bestSplitable.setSplitPoint();
            }

            return bestSplitable;
        }

        catch (Exception e) {
            e.printStackTrace();
        };

        return null;
    }

    @Override
    public double classifyInstance(Instance instance)
            throws Exception {

        double maxProbability = Double.MAX_VALUE * -1;
        double currentProbability;
        int maxIndex = 0;
        int j;

        for (j = 0; j < instance.numClasses(); j++) {
            currentProbability = getProbability(j, instance);
            if (Utils.gr(currentProbability,maxProbability)) {
                maxIndex = j;
                maxProbability = currentProbability;
            }
        }

        return (double)maxIndex;
    }

    private double getProbabilities(int classIndex, Instance instance, double weight) {
        double prob = 0;

        if(is_leaf)
        {
            return weight * nodeType.classProbability(classIndex, instance, -1);
        }
        else
        {
            int subsetIndex = nodeType.getSubsetIndex(instance);
            if(subsetIndex == -1)
            {
                double[] weights = nodeType.getWeights(instance);
                for(int i=0; i<children.length; i++)
                {
                    if(!children[i].is_empty)
                    {
                        prob += children[i].getProbabilities(classIndex, instance, weights[i]*weight);
                    }
                }
                return prob;
            }
            else
            {
                if(children[subsetIndex].is_empty)
                {
                    return weight * nodeType.classProbability(classIndex, instance, subsetIndex);
                }
                else
                {
                    return children[subsetIndex].getProbabilities(classIndex,instance,weight);
                }
            }
        }
    }

    private double getProbability(int classIndex, Instance instance) {
        return getProbabilities(classIndex, instance, 1);
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        return super.distributionForInstance(instance);
    }

    @Override
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();

        result.enable(Capabilities.Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);
        result.enable(Capabilities.Capability.DATE_ATTRIBUTES);
        result.enable(Capabilities.Capability.MISSING_VALUES);
        result.enable(Capabilities.Capability.NOMINAL_CLASS);
        result.enable(Capabilities.Capability.MISSING_CLASS_VALUES);

        result.setMinimumNumberInstances(0);

        return result;
    }

    public String toString() {

        try {
            StringBuffer text = new StringBuffer();

            if (is_leaf) {
                text.append(": ");
                text.append(nodeType.toString(0, dataSet));
            }else
                printTree(0, text);
            text.append("\n\nNumber of Leaves  : \t"+(numLeaves())+"\n");
            text.append("\nSize of the tree : \t"+numNodes()+"\n");

            return text.toString();
        } catch (Exception e) {
            return "Can't print classification tree.";
        }
    }

    public int numLeaves() {

        int num = 0;
        int i;

        if (is_leaf)
            return 1;
        else
            for (i=0;i<children.length;i++)
                num = num+children[i].numLeaves();

        return num;
    }

    public int numNodes() {

        int no = 1;
        int i;

        if (!is_leaf)
            for (i=0;i<children.length;i++)
                no = no+children[i].numNodes();

        return no;
    }

    /**
     * Print the Tree
     * @param depth
     * @param text
     * @throws Exception
     */
    private void printTree(int depth, StringBuffer text)
            throws Exception {

        int i,j;

        for (i=0;i<children.length;i++) {
            text.append("\n");;
            for (j=0;j<depth;j++)
                text.append("|   ");
            text.append(nodeType.leftSide(dataSet));
            text.append(nodeType.rightSide(i, dataSet));
            if (children[i].is_leaf) {
                text.append(": ");
                text.append(nodeType.toString(i, dataSet));
            }else
                children[i].printTree(depth + 1, text);
        }
    }

    public double getTrainingError() {
        if(is_leaf)
        {
            return nodeType.classDistribution.numIncorrect();
        }
        else
        {
            double error = 0;
            for(int i=0; i<children.length; i++)
            {
                error += children[i].getTrainingError();
            }
            return error;
        }
    }

    public static void main (String [] args) throws Exception {
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
        
        Evaluation j48eval = WekaAccess.tenFoldCrossValidation(data, new J48());
        System.out.println(j48eval.toSummaryString("-- J48 Result --\n", false));
        
        Classifier j48classifier = new J48();
        j48classifier.buildClassifier(data);
        
        System.out.println("-- J48 Model --\n" + j48classifier.toString());
        
        Evaluation myC45eval = WekaAccess.tenFoldCrossValidation(data, new myC45());
        System.out.println(myC45eval.toSummaryString("-- myC45 result --\n", false));
        
        Classifier myC45classifier = new myC45();
        myC45classifier.buildClassifier(data);

        System.out.println("-- myC45 Model --\n" + myC45classifier.toString());
        
    }
}
