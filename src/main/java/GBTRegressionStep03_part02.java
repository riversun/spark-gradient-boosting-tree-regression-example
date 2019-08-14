
import java.lang.reflect.Array;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.regression.GBTRegressor;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class GBTRegressionStep03_part02 {

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

    String[] indexedCategoricalColNames = stringIndexers
        .stream()
        .map(StringIndexer::getOutputCol)
        .toArray(String[]::new);

    String[] numericColNames = new String[] { "weight" };

    VectorAssembler assembler = new VectorAssembler()
        .setInputCols(array(indexedCategoricalColNames, numericColNames))
        .setOutputCol("features");

    GBTRegressor gbtr = new GBTRegressor()// (1)
        .setLabelCol("price")
        .setFeaturesCol("features")
        .setPredictionCol("prediction");

    PipelineStage[] indexerStages = stringIndexers.toArray(new PipelineStage[0]);

    PipelineStage[] pipelineStages = array(indexerStages, assembler, gbtr);// (2)

    Pipeline pipeline = new Pipeline().setStages(pipelineStages);

    long seed = 0;

    Dataset<Row>[] splits = dataset.randomSplit(new double[] { 0.7, 0.3 }, seed);// (3)

    Dataset<Row> trainingData = splits[0];// (4)
    Dataset<Row> testData = splits[1];// (5)

    PipelineModel pipelineModel = pipeline.fit(trainingData);// (6)

    Dataset<Row> predictions = pipelineModel.transform(testData);// (7)

    predictions.select("id", "material", "shape", "weight", "brand", "shop", "price", "prediction").show(10);// (8)

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