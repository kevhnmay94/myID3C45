/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package id3j48;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.util.Random;
import java.util.Scanner;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.Id3;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.Utils;
import weka.core.converters.CSVLoader;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;
import weka.filters.unsupervised.attribute.Remove;

public class WekaAccess {
    protected static String mainPath;
    protected static String datasetFolder;
    protected static String saveFolder;
    protected static String classifiedFolder;
    private static Scanner cin;
    
    public static void initializePath(){
        mainPath = new File("").getAbsolutePath();
        mainPath += File.separator;
        datasetFolder = mainPath + "dataset";
        saveFolder = mainPath + "result";
        classifiedFolder = mainPath + "classified";
        System.out.println(new File("").getAbsolutePath());
    }

    public static Instances readArff(String filename) throws Exception {
        ConverterUtils.DataSource source = new ConverterUtils.DataSource(datasetFolder+File.separator+filename);
        Instances data = source.getDataSet();
        if (data.classIndex() == -1)
                   data.setClassIndex(data.numAttributes() - 1);
        return data;
    }
    
    public static Instances readCsv(String filename) throws Exception{
        CSVLoader csvLoader = new CSVLoader();
            csvLoader.setSource(new File(datasetFolder + File.separator + filename));
            Instances data = csvLoader.getDataSet();
            if(data.classIndex() == -1)
            {
                data.setClassIndex(data.numAttributes()-1);
            }
            return data;
    }
    
    public static Instances removeAttribute(Instances data, int index) throws Exception{
            String options = "-R "+ String.valueOf(index) ;
            Remove remove = new Remove();
            remove.setOptions(Utils.splitOptions(options));
            remove.setInputFormat(data);
            Instances newDataSet = Filter.useFilter(data,remove);
            return newDataSet;
    }
    
    public static Instances resampleData(Instances data) throws Exception{
        Resample resample = new Resample();
        String filterOptions = "-B 0.0 -S 1 -Z 100.0";
        resample.setOptions(Utils.splitOptions(filterOptions));
        resample.setRandomSeed(1);
        resample.setInputFormat(data);
        Instances newDataSet = Filter.useFilter(data,resample);
        return newDataSet;
    }
    
    public static Classifier buildClassifier(Instances data, Classifier classifier) throws Exception
    {
        classifier.buildClassifier(data);
        return classifier;
    }
    
    public static Evaluation testModel(Classifier classifier, Instances data, Instances test) throws Exception{
        Evaluation evaluation = new Evaluation(data);
        evaluation.evaluateModel(classifier, test);
        return evaluation;
    }
    
    public static Evaluation tenFoldCrossValidation(Instances data, Classifier classifier) throws Exception{
        Evaluation eval = new Evaluation(data);
        eval.crossValidateModel(classifier, data, 10, new Random(1));
        return eval;
    }
    
    public static Evaluation percentageSplit(Instances data, Classifier classifier, int percentage) throws Exception{
        Instances tempdata = new Instances(data);
        tempdata.randomize(new Random(1));

        int trainSize = Math.round(tempdata.numInstances() * percentage / 100);
        int testSize = tempdata.numInstances() - trainSize;
        Instances train = new Instances(tempdata, 0, trainSize);
        Instances test = new Instances(tempdata, trainSize, testSize);

        classifier.buildClassifier(train);
        Evaluation eval = testModel(classifier, train, test);
        return eval;
    }
    
    public static void saveModel(String filename, Classifier classifier) throws Exception{
        SerializationHelper.write(saveFolder + File.separator + filename, classifier);
    }
    
    public static Classifier loadModel(String filename) throws Exception{
        Classifier classifier = (Classifier) SerializationHelper.read(saveFolder + File.separator + filename);
        return classifier;
    }
    
    public static void classify(String filename, Classifier classifier) throws Exception{
        Instances input = readArff(filename);
        input.setClassIndex(input.numAttributes()-1);
        for(int i=0; i<input.numInstances(); i++)
        {
            double classLabel = classifier.classifyInstance(input.instance(i));
            input.instance(i).setClassValue(classLabel);
            System.out.println("Instance: " + input.instance(i));
            System.out.println("Class: " + input.classAttribute().value((int)classLabel));
        }

        try (BufferedWriter writer = new BufferedWriter(
                new FileWriter(classifiedFolder + File.separator + filename))) {
            writer.write(input.toString());
            writer.newLine();
            writer.flush();
        }
    }

    public static void main(String [] args) {
        initializePath();
        try {
            cin = new Scanner(System.in);
            Instances data = null, tempdata;
            Classifier NBclassifier, ID3classifier, j48classifier;
            Evaluation NBeval, ID3eval, j48eval;
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
            System.out.println(data.toString());
            System.out.println("Resample data? (y for yes) ");
            String resample = cin.nextLine();
            if(resample.equalsIgnoreCase("y")){
                try {
                    tempdata = resampleData(data);
                    System.out.println("-- Resampled data --");
                    System.out.println(tempdata.toString());
                } catch (Exception ex) {
                    Logger.getLogger(WekaAccess.class.getName()).log(Level.SEVERE, null, ex);
                }
            }   tempdata = removeAttribute(data, data.numAttributes());
            System.out.println("-- Remove Attribute --");
            System.out.println(tempdata.toString());
            NBclassifier = buildClassifier(data, new NaiveBayes());
            System.out.println("-- Naive Bayes Classifier --");
            System.out.println(NBclassifier.toString());
            ID3classifier = buildClassifier(data, new Id3());
            System.out.println("-- ID3 Classifier --");
            System.out.println(ID3classifier.toString());
            j48classifier = buildClassifier(data, new J48());
            System.out.println("-- J48 Classifier --");
            System.out.println(j48classifier.toString());
            Instances test = null;
            if(extension.equalsIgnoreCase("arff"))
                test = readArff("test." + filename);
            else if(extension.equalsIgnoreCase("csv"))
                test = readCsv("test." + filename);
            NBeval = testModel(NBclassifier, data, test);
            System.out.println(NBeval.toSummaryString("-- Training set evaluation results with Naive Bayes --\n", false));
            ID3eval = testModel(ID3classifier, data, test);
            System.out.println(NBeval.toSummaryString("-- Training set evaluation results with ID3 --\n", false));
            j48eval = testModel(j48classifier, data, test);
            System.out.println(NBeval.toSummaryString("-- Training set evaluation results with J48 --\n", false));
            NBeval = tenFoldCrossValidation(data, NBclassifier);
            System.out.println(NBeval.toSummaryString("-- 10-fold cross validation results with Naive Bayes --\n", false));
            ID3eval = tenFoldCrossValidation(data, ID3classifier);
            System.out.println(NBeval.toSummaryString("-- 10-fold cross validation results with ID3 --\n", false));
            j48eval = tenFoldCrossValidation(data, j48classifier);
            System.out.println(NBeval.toSummaryString("-- 10-fold cross validation results with J48 --\n", false));
            NBeval = percentageSplit(data, NBclassifier,66);
            System.out.println(NBeval.toSummaryString("-- 66% split validation results with Naive Bayes --\n", false));
            ID3eval = percentageSplit(data, ID3classifier,66);
            System.out.println(NBeval.toSummaryString("-- 66% split validation results with ID3 --\n", false));
            j48eval = percentageSplit(data, j48classifier,66);
            System.out.println(NBeval.toSummaryString("-- 66% split validation results with J48 --\n", false));
            System.out.println("-- Save Naive Bayes Model --");
            saveModel("nb."+name+".model", NBclassifier);
            System.out.println("-- Save Naive Bayes Model --");
            saveModel("id3."+name+".model", ID3classifier);
            System.out.println("-- Save Naive Bayes Model --");
            saveModel("j48."+name+".model", j48classifier);
            System.out.println("-- Save Naive Bayes Model --");
            saveModel("nb."+name+".model", NBclassifier);
            System.out.println("-- Save ID3 Model --");
            saveModel("id3."+name+".model", ID3classifier);
            System.out.println("-- Save J48 Model --");
            saveModel("j48."+name+".model", j48classifier);
            System.out.println("-- Load Naive Bayes Model --");
            System.out.println(loadModel("nb."+name+".model").toString());
            System.out.println("-- Load ID3 Model --");
            System.out.println(loadModel("id3."+name+".model").toString());
            System.out.println("-- Load J48 Model --");
            System.out.println(loadModel("j48."+name+".model").toString());
            System.out.println("-- Classify Naive Bayes Model --");
            classify("classify."+filename,NBclassifier);
            System.out.println("-- Classify ID3 Model --");
            classify("classify."+filename,ID3classifier);
            System.out.println("-- Classify J48 Model --");
            classify("classify."+filename,j48classifier);
        } catch (Exception ex) {
            Logger.getLogger(WekaAccess.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
}
