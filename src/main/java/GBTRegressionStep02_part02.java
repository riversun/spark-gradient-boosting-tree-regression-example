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
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class GBTRegressionStep02_part02 {

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
        .load("dataset/gem_price.csv");// gem_price_ja.csv for Japanese

    StringIndexer materialIndexer = new StringIndexer()// (1)
        .setInputCol("material")
        .setOutputCol("materialIndex");

    StringIndexer shapeIndexer = new StringIndexer()// (2)
        .setInputCol("shape")
        .setOutputCol("shapeIndex");

    StringIndexer brandIndexer = new StringIndexer()// (3)
        .setInputCol("brand")
        .setOutputCol("brandIndex");

    StringIndexer shopIndexer = new StringIndexer()// (4)
        .setInputCol("shop")
        .setOutputCol("shopIndex");

    Dataset<Row> dataset1 = materialIndexer.fit(dataset).transform(dataset);// (5)

    Dataset<Row> dataset2 = shapeIndexer.fit(dataset).transform(dataset1);// (6)

    Dataset<Row> dataset3 = brandIndexer.fit(dataset).transform(dataset2);// (7)

    Dataset<Row> dataset4 = shopIndexer.fit(dataset).transform(dataset3);// (8)

    dataset4.show(10);

  }

}