package com.gmail.mbotyuk

import org.apache.log4j.{Level, LogManager}
import org.apache.spark.sql.types.{FloatType, IntegerType, StringType, StructType}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, StringIndexerModel, VectorAssembler}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.{Column, DataFrame, Dataset, Row, SparkSession}

object Titanic {

  private val log = LogManager.getRootLogger

  def main(args: Array[String]): Unit = {

    val spark = SparkSession
      .builder()
      .master("local[*]")
      .appName("Titanic")
      .getOrCreate()

    log.setLevel(Level.WARN)
    spark.sparkContext.setLogLevel("WARN")
    spark.conf.set("spark.sql.session.timeZone", "UTC")

    log.warn("Read file from path: " + args(0))
    val dfTrain = read(spark, args(0))
    log.warn("Read OK")

    // наглядно увидеть данные
    dfTrain.cache().show(100)

    // Начало подготовки и анализа

    val dfPrepared = dataPreparation(dfTrain)
    analysis(dfPrepared, spark)

    val Array(trainingData, validationData) = dfPrepared.randomSplit(Array(0.7, 0.3))

    // Конец подготовки и анализа

    // ML
    log.warn("Start ML analysis. Waiting...")
    val catFeatColNames = Seq("Pclass", "Sex", "Embarked")
    val numFeatColNames = Seq("Age", "SibSp", "Parch", "Fare")

    val idxdCatFeatColName = catFeatColNames.map(_ + "Indexed")
    val allIdxdFeatColNames = numFeatColNames ++ idxdCatFeatColName
    val assembler = new VectorAssembler()
      .setInputCols(Array(allIdxdFeatColNames: _*))
      .setOutputCol("Features")

    def indexer(colName: String, ds: Dataset[Row]): StringIndexerModel = {
      new StringIndexer()
        .setInputCol(colName)
        .setOutputCol(colName + "Indexed")
        .fit(ds)
    }

    val stringIndexers = catFeatColNames.map { colName =>
      indexer(colName, trainingData)
    }
    val labelIndexer = indexer("Survived", trainingData)

    def classifier(label: String): RandomForestClassifier = {
      new RandomForestClassifier()
        .setLabelCol(label + "Indexed")
        .setFeaturesCol("Features")
    }

    val randomForest = classifier("Survived")

    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(labelIndexer.labels)

    val pipeline = new Pipeline().setStages(
      (stringIndexers :+ labelIndexer :+ assembler :+ randomForest :+ labelConverter).toArray)

    val evaluator = new BinaryClassificationEvaluator()
      .setLabelCol("SurvivedIndexed")
      .setMetricName("areaUnderPR")
    val paramGrid = new ParamGridBuilder()
      .addGrid(randomForest.maxBins, Array(25, 28, 31))
      .addGrid(randomForest.maxDepth, Array(4, 6, 8))
      .addGrid(randomForest.impurity, Array("entropy", "gini"))
      .build()
    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(10)

    val crossValidatorModel = cv.fit(trainingData)
    val predictions = crossValidatorModel.transform(validationData)

    // получаем результат работы
    printResult(evaluator, predictions, crossValidatorModel, dfPrepared)

    log.warn("Write file to path: " + args(1))
    write(predictions, args(1))
    log.warn("Write OK")

    spark.close()
  }

  private def read(spark: SparkSession, pathRead: String): DataFrame = {
    val dataScheme_train = (new StructType)
      .add("PassengerId", IntegerType)
      .add("Survived", IntegerType)
      .add("Pclass", IntegerType)
      .add("Name", StringType)
      .add("Sex", StringType)
      .add("Age", FloatType)
      .add("SibSp", IntegerType)
      .add("Parch", IntegerType)
      .add("Ticket", StringType)
      .add("Fare", FloatType)
      .add("Cabin", StringType)
      .add("Embarked", StringType)

    val trainSchema = StructType(dataScheme_train)
    spark.read.format("csv").option("header", "true").schema(trainSchema).load(pathRead)
  }

  private def dataPreparation(df: DataFrame): DataFrame = {
    // поиск пустых значений
    def countCols(columns: Array[String]): Array[Column] = {
      columns.map(c => {
        count(when(col(c).isNull ||
          col(c) === "" ||
          col(c).contains("NULL") ||
          col(c).contains("null"), c)
        ).alias(c)
      })
    }

    df.select(countCols(df.columns): _*).show()

    // в столбце Cabin много пустых значений - удаляю
    // столбцы Name и Ticket бесполезны для анализа - удаляю
    val dfDropCol = df.drop("Cabin").drop("Name").drop("Ticket")

    // в столбце Age есть пустые значения - заполняю средним значением
    val dfAvgAge = dfDropCol.agg(mean(dfDropCol("Age"))).first.getDouble(0)
    val dfNotNullAge = dfDropCol.na.fill(dfAvgAge, Array("Age"))

    // значение S в столбце Embarked встречается чаще всего - им и заполню пустоты
    val embarked: (String => String) = {
      case "" => "S"
      case null => "S"
      case "null" => "S"
      case a => a
    }
    val embarkedUDF = udf(embarked)
    val dfPrepared = dfNotNullAge.withColumn("Embarked", embarkedUDF(dfNotNullAge.col("Embarked")))

    dfPrepared.select(countCols(dfPrepared.columns): _*).show()
    dfPrepared
  }

  private def analysis(df: DataFrame, spark: SparkSession): Unit = {
    // поиск вылетов, вижу вылеты в столбцах Age (в среднем 29 и максимальный 80), SibSp (в среднем 0-1 и максимальный 8), Parch (в среднем 0-1 и максимальный 6), Fare (в среднем 32 и максимальный 512)
    val numericColumns: Array[String] = df.dtypes.filter(p => p._2.equals("IntegerType") || p._2.equals("FloatType")).map(_._1)
    df.select(numericColumns.map(col).toIndexedSeq: _*).summary().show()

    // проверка сбалансированности данных - разница не большая
    df.groupBy("Survived").count().show()

    df.describe("Pclass", "Age", "SibSp", "Parch", "Fare").show()

    // получение статистики для презентации
    import spark.implicits._
    val df_group_pclass = df.groupBy($"Pclass").count().orderBy($"Pclass")

    df.groupBy($"Pclass", $"Survived").count()
      .orderBy($"Pclass").withColumnRenamed("count", "order_by_count")
      .join(df_group_pclass, Seq("Pclass"))
      .filter($"Survived" === 1)
      .withColumn("perc_of_count_total", ($"order_by_count" * lit(100) / $"count"))
      .show()

    df.filter($"Survived" === 1)
      .groupBy($"Pclass").count()
      .orderBy($"Pclass")
      .withColumn("perc_of_count_total", $"count" / lit(891) * 100)
      .show()

    df.select($"Survived", $"Age")
      .groupBy($"Survived", $"Age").count()
      .orderBy($"Age")
      .show(891)

    df.createOrReplaceTempView("df_train")
    spark.sql("select Embarked,count(*) from df_train group by Embarked").show()
    spark.sql("select Sex, Survived, count(*) from df_train group by Sex,Survived").show()
    spark.sql("select Pclass, Sex, count(*) from df_train group by Pclass,Sex").show()
    spark.sql("select Pclass, Survived, Sex, count(*) from df_train group by Pclass,Survived,Sex").show()
    spark.sql("select SibSp, Survived, count(*) from df_train group by SibSp,Survived order by SibSp").show()
    spark.sql("select Parch, Survived, count(*) from df_train group by Parch,Survived order by Parch").show()
  }

  private def printResult(evaluator: BinaryClassificationEvaluator, predictions: DataFrame, crossValidatorModel: CrossValidatorModel, dfPrepared: DataFrame): Unit = {
    val accuracy = evaluator.evaluate(predictions)
    log.warn("Test Error DT = " + (1.0 - accuracy))

    def accuracyScore(df: DataFrame, label: String, predictCol: String) = {
      val rdd = df.select(label, predictCol).rdd.map(row ⇒ (row.getInt(0).toDouble, row.getDouble(1)))
      new MulticlassMetrics(rdd).accuracy
    }

    log.warn("train accuracy with pipeline " + accuracyScore(crossValidatorModel.transform(dfPrepared), "Survived", "prediction"))
  }

  private def write(df: DataFrame, pathWrite: String): Unit = {
    df
      .withColumn("Survived", col("predictedLabel"))
      .select("PassengerId", "Survived")
      .coalesce(1)
      .write
      .mode("overwrite")
      .format("csv")
      .option("header", "true")
      .save(pathWrite)
  }
}
