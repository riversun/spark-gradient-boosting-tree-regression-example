
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
        .load("dataset/gem_price.csv");// (6)// gem_price_ja.csv for Japanese

    dataset.show();// (7)

    dataset.printSchema();// (8)

  }

}