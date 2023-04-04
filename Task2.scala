package edu.ucr.cs.cs167.tagra002

import org.apache.spark.SparkConf
import org.apache.spark.beast.SparkSQLRegistration
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession}
// Daniel Boules
object Task2 {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("Beast Example")
    if (!conf.contains("spark.master"))
      conf.setMaster("local[*]")
    val spark: SparkSession.Builder = SparkSession.builder().config(conf)
    val sparkSession: SparkSession = spark.getOrCreate()
    SparkSQLRegistration.registerUDT
    SparkSQLRegistration.registerUDF(sparkSession)

    val inputFile: String = args(0)
    try {
      // Read the JSON data from the input file and create a DataFrame
      val input = sparkSession.read.format("json")
        .option("sep", "\t")
        .option("inferSchema", "true")
        .option("header", "true")
        .load(inputFile)
      // Get the top 20 hashtags from the DataFrame
      val topHashtags: Seq[String] = input
        .select(explode(col("hashtags")).as("hashtag"))
        .groupBy("hashtag")
        .count()
        .sort(desc("count"))
        .limit(20)
        .select("hashtag")
        .collect()
        .map(_.getString(0))
        .toSeq

      //// Find the tweets that contain at least one of the top hashtags
      val withTopic: DataFrame = input
        .withColumn("topic", array_intersect(col("hashtags"), typedLit(topHashtags)))
        .withColumn("topic", element_at(col("topic"), 1))
        .drop("hashtags")
        .filter(col("topic").isNotNull)
      withTopic.coalesce(1).write.mode(SaveMode.Overwrite).json("tweets_topic.json")
    } finally {
      // Stop the SparkSession
      sparkSession.stop()
    }
  }
}