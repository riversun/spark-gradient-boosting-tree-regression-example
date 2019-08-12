
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class GBTRegressionStep02_part01 {

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

    StringIndexer materialIndexer = new StringIndexer()// (1)
        .setInputCol("material")// (2)
        .setOutputCol("materialIndex");// (3)

    Dataset<Row> materialIndexAddedDataSet = materialIndexer.fit(dataset).transform(dataset);// (4)

    materialIndexAddedDataSet.show(10);// (5)

  }

}