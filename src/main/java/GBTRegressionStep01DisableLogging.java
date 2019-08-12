
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class GBTRegressionStep01DisableLogging {

  public static void main(String[] args) {

    org.apache.log4j.Logger.getLogger("org").setLevel(org.apache.log4j.Level.ERROR);
    org.apache.log4j.Logger.getLogger("akka").setLevel(org.apache.log4j.Level.ERROR);

    SparkSession spark = SparkSession
        .builder()
        .appName("GradientBoostingTreeGegression")
        .master("local[*]")// (1)
        .getOrCreate();

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