
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class GBTRegressionStep02_part03 {

  public static void main(String[] args) {

    //System.setProperty("hadoop.home.dir", "c:\\Temp\\winutil\\");//for windows
    
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

    StringIndexer materialIndexer = new StringIndexer()
        .setInputCol("material")
        .setOutputCol("materialIndex");

    StringIndexer shapeIndexer = new StringIndexer()
        .setInputCol("shape")
        .setOutputCol("shapeIndex");

    StringIndexer brandIndexer = new StringIndexer()
        .setInputCol("brand")
        .setOutputCol("brandIndex");

    StringIndexer shopIndexer = new StringIndexer()
        .setInputCol("shop")
        .setOutputCol("shopIndex");

    Pipeline pipeline = new Pipeline()
        .setStages(new PipelineStage[] { materialIndexer, shapeIndexer, brandIndexer, shopIndexer });// (1)

    PipelineModel pipelineModel = pipeline.fit(dataset);// (2)

    Dataset<Row> indexedDataset = pipelineModel.transform(dataset);// (3)
    indexedDataset.show(10);

  }

}