
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class GBTRegressionStep01ForWindows {

  public static void main(String[] args) {

    System.setProperty("hadoop.home.dir", "c:\\Temp\\winutil\\");
    
    SparkSession spark = SparkSession
        .builder()
        .appName("GradientBoostingTreeGegression")
        .master("local[*]")// (1)
        .getOrCreate();

    spark.sparkContext().setLogLevel("OFF");// (2)

    Dataset<Row> dataset = spark
        .read()
        .format("csv")// (3)
        .option("header", "true")// (4)
        .option("inferSchema", "true")// (5)
        .load("dataset/gem_price_ja.csv");// (6)

    dataset.show();// (7)

    dataset.printSchema();// (8)

  }

}