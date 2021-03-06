
/* 
 *  Copyright (c) 2019 Tom Misawa, riversun.org@gmail.com
 *  
 *  Permission is hereby granted, free of charge, to any person obtaining a
 *  copy of this software and associated documentation files (the "Software"),
 *  to deal in the Software without restriction, including without limitation
 *  the rights to use, copy, modify, merge, publish, distribute, sublicense,
 *  and/or sell copies of the Software, and to permit persons to whom the
 *  Software is furnished to do so, subject to the following conditions:
 *  
 *  The above copyright notice and this permission notice shall be included in
 *  all copies or substantial portions of the Software.
 *  
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 *  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 *  DEALINGS IN THE SOFTWARE.
 *  
 */
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class GBTRegressionStep02_part03 {

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
        .load("dataset/gem_price.csv");// gem_price_ja.csv for Japanese

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