import evaluationMetric.MyMetric;
import org.grouplens.lenskit.ItemScorer;
import org.grouplens.lenskit.baseline.BaselineScorer;
import org.grouplens.lenskit.baseline.ItemMeanRatingItemScorer;
import org.grouplens.lenskit.baseline.UserMeanBaseline;
import org.grouplens.lenskit.baseline.UserMeanItemScorer;
import org.grouplens.lenskit.core.LenskitConfiguration;
import org.grouplens.lenskit.data.source.DataSource;
import org.grouplens.lenskit.data.source.GenericDataSource;
import org.grouplens.lenskit.data.text.DelimitedColumnEventFormat;
import org.grouplens.lenskit.data.text.EventFormat;
import org.grouplens.lenskit.data.text.RatingEventType;
import org.grouplens.lenskit.data.text.TextEventDAO;
import org.grouplens.lenskit.eval.algorithm.AlgorithmInstance;
import org.grouplens.lenskit.eval.data.crossfold.CrossfoldTask;
import org.grouplens.lenskit.eval.metrics.predict.NDCGPredictMetric;
import org.grouplens.lenskit.eval.traintest.SimpleEvaluator;
import org.grouplens.lenskit.iterative.IterationCount;
import org.grouplens.lenskit.knn.item.ItemItemScorer;
import org.grouplens.lenskit.mf.funksvd.FeatureCount;
import org.grouplens.lenskit.mf.funksvd.FunkSVDItemScorer;
import org.grouplens.lenskit.transform.normalize.BaselineSubtractingUserVectorNormalizer;
import org.grouplens.lenskit.transform.normalize.UserVectorNormalizer;

import java.io.File;
import java.util.Random;

public class Runner1 {
	public static void main(String args[]) throws Exception {
		LenskitConfiguration config2 = new LenskitConfiguration();
		config2.bind(ItemScorer.class).to(FunkSVDItemScorer.class);
		config2.bind(BaselineScorer.class, ItemScorer.class).to(UserMeanItemScorer.class);
		config2.bind(UserMeanBaseline.class, ItemScorer.class).to(ItemMeanRatingItemScorer.class);
		//config.bind(UserVectorNormalizer.class).to(BaselineSubtractingUserVectorNormalizer.class);
		config2.set(FeatureCount.class).to(4);
		config2.set(IterationCount.class).to(1000);
		/*EventFormat eventFormat = new DelimitedColumnEventFormat(new RatingEventType());
		//config.bind(EventDAO.class).to(new TextEventDAO(new File("D:\\bigdata\\movielens\\ml-100k\\u.data"), eventFormat));
		//config.bind(EventDAO.class).to(new SimpleFileRatingDAO(new File("D:\\bigdata\\movielens\\ml-1m\\ratings.dat"), "::"));
		LenskitRecommender rec = LenskitRecommender.build(config);
		ItemRecommender irec = rec.getItemRecommender();
		for (int i = 1; i <= 10; i++) {
			List<ScoredId> list = irec.recommend(i, 2);
			System.out.println(i + " " + list);
		}*/

		EventFormat eventFormat = new DelimitedColumnEventFormat(new RatingEventType());

		LenskitConfiguration config = new LenskitConfiguration();
// Use item-item CF to score items
		config.bind(ItemScorer.class)
				.to(ItemItemScorer.class);
// let's use personalized mean rating as the baseline/fallback predictor.
// 2-step process:
// First, use the user mean rating as the baseline scorer
		config.bind(BaselineScorer.class, ItemScorer.class)
				.to(UserMeanItemScorer.class);
// Second, use the item mean rating as the base for user means
		config.bind(UserMeanBaseline.class, ItemScorer.class)
				.to(ItemMeanRatingItemScorer.class);
// and normalize ratings by baseline prior to computing similarities
		config.bind(UserVectorNormalizer.class)
				.to(BaselineSubtractingUserVectorNormalizer.class);
		//config.bind(EventDAO.class).to(new SimpleFileRatingDAO(new File("ratings.csv"), ","));


		SimpleEvaluator evaluator = new SimpleEvaluator();

		NDCGPredictMetric metric = new NDCGPredictMetric();
		//evaluator.addMetric(metric);
		evaluator.addMetric(new MyMetric());

		AlgorithmInstance instance = new AlgorithmInstance("yo1", config);
		instance.setRandom(new Random());
		evaluator.addAlgorithm(instance);
		//evaluator.addAlgorithm(new AlgorithmInstance("sdsa", config2));
		//DataSource dataSource = new GenericDataSource("fileName10", new TextEventDAO(new File("D:\\bigdata\\movielens\\ml-100k\\u.data"), eventFormat));
		DataSource dataSource = new GenericDataSource("file10", new TextEventDAO(new File("D:\\bigdata\\movielens\\fake\\all_ratings"), eventFormat));
		//DataSource dataSource2 = new GenericDataSource("file2", new TextEventDAO(new File("D:\\bigdata\\movielens\\fake\\test"), eventFormat));

		evaluator.setOutputPath("res1.txt");
		evaluator.setUserOutputPath("SD");
		evaluator.setPredictOutputPath("pred");

		CrossfoldTask task = new CrossfoldTask("task");
		//task.setHoldout(3);
		task.setPartitions(2);
		task.setSource(dataSource);
		evaluator.addDataset(task);
		//evaluator.addDataset(dataSource, 2);
		//evaluator.addDataset(dataSource, dataSource2);
		evaluator.call();

		/*TrainTestEvalTask task = new TrainTestEvalTask("Eval name");
		AlgorithmInstance instance = new AlgorithmInstance("alg1", config);
		task.addAlgorithm(instance);
		GenericTTDataBuilder builder = new GenericTTDataBuilder("builder");
		builder.setName("builder name");
		builder.setTest(new PackedDataSource("train", new File("D:\\bigdata\\movielens\\ml-100k\\u.data"), null));
		builder.setTrain(new PackedDataSource("train", new File("D:\\bigdata\\movielens\\ml-100k\\u.data"), null));
		GenericTTDataSet dataSet = builder.build();
		dataSet.configure(config);
		task.addDataset(dataSet);
		task.perform();*/
	}
}
