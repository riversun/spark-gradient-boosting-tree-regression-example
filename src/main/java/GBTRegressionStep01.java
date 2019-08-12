
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

/**
 * STEP 01 Read dataset from CSV file
 * 
 * @author tom
 *
 */
public class GBTRegressionStep01 {

  public static void main(String[] args) {

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