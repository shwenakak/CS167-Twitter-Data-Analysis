package edu.ucr.cs.cs167.tagra002

import org.apache.spark.SparkConf
import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator}
import org.apache.spark.ml.feature.{HashingTF, StringIndexer, Tokenizer}
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit, TrainValidationSplitModel}
// Shwena Kak
object Task3 {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("Beast Example")
    if (!conf.contains("spark.master"))
      conf.setMaster("local[*]")
    val spark: SparkSession.Builder = SparkSession.builder().config(conf)
    val sparkSession: SparkSession = spark.getOrCreate()

    val inputFile: String = args(0)
//    val modelFile: String = args(1)

    try {
      val tweetsDF: DataFrame = sparkSession.read.format("json")
        .option("sep", "\t")
        .option("inferSchema", "true")
        .option("header", "true")
        .load(inputFile)

      val tokenizer = new Tokenizer()
        .setInputCol("text")
        .setOutputCol("words")

      val hashingTF = new HashingTF()
        .setInputCol("words")
        .setOutputCol("features")

      val stringIndexer = new StringIndexer()
        .setInputCol("topic")
        .setOutputCol("label")
        .setHandleInvalid("skip")

      val logisticRegression = new LogisticRegression()

      val pipeline = new Pipeline().setStages(Array(tokenizer, hashingTF, stringIndexer, logisticRegression))

      val paramGrid = new ParamGridBuilder()
        .addGrid(hashingTF.numFeatures, Array(10, 100, 1000))
        .addGrid(logisticRegression.regParam, Array(0.01, 0.1, 0.3, 0.8))
        .build()

      val cv = new TrainValidationSplit()
        .setEstimator(pipeline)
        .setEvaluator(new MulticlassClassificationEvaluator)
        .setEstimatorParamMaps(paramGrid)
        .setTrainRatio(0.8)
        .setParallelism(2)

      val Array(trainingData, testData) = tweetsDF.randomSplit(Array(0.8, 0.2))

      val logisticModel: TrainValidationSplitModel = cv.fit(trainingData)

      val predictions = logisticModel.transform(testData)
      predictions.select("id", "text", "topic", "user_description", "label", "prediction").show()

      val binaryClassificationEvaluator = new BinaryClassificationEvaluator()
        .setLabelCol("label")
        .setRawPredictionCol("prediction")

      val precision = binaryClassificationEvaluator.evaluate(predictions)
      println(s"Overall Precision: $precision ")

      val recallEvaluator = new MulticlassClassificationEvaluator()
        .setLabelCol("label")
        .setPredictionCol("prediction")
        .setMetricName("weightedRecall")

      val recall = recallEvaluator.evaluate(predictions)
      println(s"Overall Recall $recall")

      predictions.select("id", "text", "topic", "user_description", "label", "prediction")
        .coalesce(1)
        .write
        .mode(SaveMode.Overwrite)
        .json("tweets_predicted.json")
    } finally{
      sparkSession.stop()
    }
  }
}