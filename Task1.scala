package edu.ucr.cs.cs167.tagra002

import org.apache.spark.SparkConf
import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession}
// Trisha Agrawal
object Task1 {
  def main(args: Array[String]): Unit = {
    // set up used from lab 9
    val conf = new SparkConf().setAppName("Beast Example")
    if (!conf.contains("spark.master"))
      conf.setMaster("local[*]")
    val spark: SparkSession.Builder = SparkSession.builder().config(conf)
    val sparkSession: SparkSession = spark.getOrCreate()
    // input file argument
    val inputFile: String = args(0)
    try {
      // read in file in json format
      val input = sparkSession.read.format("json")
        .option("sep", "\t")
        .option("inferSchema", "true")
        .option("header", "true")
        .load(inputFile)
      val tweetsDF: DataFrame = input
      // selecting needed attributes only
      val newDF: DataFrame = tweetsDF.selectExpr("id","text",
        "entities.hashtags.text as hashtags",
        "user.description as user_description",
        "retweet_count","reply_count","quoted_status_id")
      // write cleaned data set to new json file
      newDF.coalesce(1).write.mode(SaveMode.Overwrite).json("tweets_clean.json")
      // create list of hashtags only
      val exDF: DataFrame = newDF.selectExpr("EXPLODE(hashtags) as hashtags")
      exDF.createOrReplaceTempView("exDF")
      // use SQL query to find top 20 hashtags
      sparkSession.sql(s"""
          SELECT hashtags, count(*) as counts
          FROM exDF
          GROUP BY hashtags
          ORDER BY counts DESC
          LIMIT 20
          """).foreach(row => println(s"${row.get(0)}\t${row.get(1)}"))
    } finally {
      sparkSession.stop()
    }
  }
}