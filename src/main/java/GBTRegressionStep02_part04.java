
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class GBTRegressionStep02_part04 {

  public static void main(String[] args) {

    // System.setProperty("hadoop.home.dir", "c:\\Temp\\winutil\\");//for windows

    org.apache.log4j.Logger.getLogger("org").setLevel(org.apache.log4j.Level.ERROR);
    org.apache.log4j.Logger.getLogger("akka").setLevel(org.apache.log4j.Level.ERROR);

    SparkSession spark = SparkSession
        .builder()
        .appName("GradientBoostingTreeGegression")
        .master("local[*]")
        .getOrCreate();

    Dataset<Row> dataset = spark
        .read()
        .format("csv")
        .option("header", "true")
        .option("inferSchema", "true")
        .load("dataset/gem_price_ja.csv");

    List<String> categoricalColNames = Arrays.asList("material", "shape", "brand", "shop");// (1)

    List<StringIndexer> stringIndexers = categoricalColNames.stream()
        .map(col -> new StringIndexer()
            .setStringOrderType("frequencyDesc")
            .setInputCol(col)
            .setOutputCol(col + "Index"))// (2)

        .collect(Collectors.toList());

    Pipeline pipeline = new Pipeline()
        .setStages(stringIndexers.toArray(new PipelineStage[0]));// (3)

    PipelineModel pipelineModel = pipeline.fit(dataset);

    Dataset<Row> indexedDataset = pipelineModel.transform(dataset);
    indexedDataset.show(10);

  }

}