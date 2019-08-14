
import java.lang.reflect.Array;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class GBTRegressionStep03_part01 {

  public static void main(String[] args) {

    System.setProperty("hadoop.home.dir", "c:\\Temp\\winutil\\");// for windows

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

    List<String> categoricalColNames = Arrays.asList("material", "shape", "brand", "shop");

    List<StringIndexer> stringIndexers = categoricalColNames.stream()
        .map(col -> new StringIndexer()
            .setStringOrderType("frequencyDesc")
            .setInputCol(col)
            .setOutputCol(col + "Index"))
        .collect(Collectors.toList());

    String[] indexedCategoricalColNames = stringIndexers// (1)
        .stream()
        .map(StringIndexer::getOutputCol)
        .toArray(String[]::new);

    String[] numericColNames = new String[] { "weight" };// (2)

    VectorAssembler assembler = new VectorAssembler()// (3)
        .setInputCols(array(indexedCategoricalColNames, numericColNames))
        .setOutputCol("features");

    PipelineStage[] indexerStages = stringIndexers.toArray(new PipelineStage[0]);// (5)

    PipelineStage[] pipelineStages = array(indexerStages, assembler);// (6)

    Pipeline pipeline = new Pipeline().setStages(pipelineStages);// (7)

    pipeline.fit(dataset).transform(dataset).show(10);// (8)

  }

  @SuppressWarnings("unchecked")
  public static <T> T[] array(final T[] array1, final T... array2) {
    final Class<?> type1 = array1.getClass().getComponentType();
    final T[] joinedArray = (T[]) Array.newInstance(type1, array1.length + array2.length);
    System.arraycopy(array1, 0, joinedArray, 0, array1.length);
    System.arraycopy(array2, 0, joinedArray, array1.length, array2.length);
    return joinedArray;
  }
}