import evaluationMetric.ExampleMetric;
import org.grouplens.lenskit.ItemRecommender;
import org.grouplens.lenskit.ItemScorer;
import org.grouplens.lenskit.baseline.BaselineScorer;
import org.grouplens.lenskit.baseline.ItemMeanRatingItemScorer;
import org.grouplens.lenskit.baseline.UserMeanBaseline;
import org.grouplens.lenskit.baseline.UserMeanItemScorer;
import org.grouplens.lenskit.core.LenskitConfiguration;
import org.grouplens.lenskit.core.LenskitRecommender;
import org.grouplens.lenskit.data.dao.EventDAO;
import org.grouplens.lenskit.data.dao.SimpleFileRatingDAO;
import org.grouplens.lenskit.data.dao.packed.RatingSnapshotDAO;
import org.grouplens.lenskit.data.history.RatingVectorUserHistorySummarizer;
import org.grouplens.lenskit.data.source.DataSource;
import org.grouplens.lenskit.data.source.GenericDataSource;
import org.grouplens.lenskit.data.text.*;
import org.grouplens.lenskit.eval.algorithm.AlgorithmInstance;
import org.grouplens.lenskit.eval.data.crossfold.CrossfoldTask;
import org.grouplens.lenskit.eval.metrics.predict.NDCGPredictMetric;
import org.grouplens.lenskit.eval.metrics.predict.RMSEPredictMetric;
import org.grouplens.lenskit.eval.metrics.topn.ItemSelectors;
import org.grouplens.lenskit.eval.metrics.topn.MAPTopNMetric;
import org.grouplens.lenskit.eval.metrics.topn.NDCGTopNMetric;
import org.grouplens.lenskit.eval.traintest.SimpleEvaluator;
import org.grouplens.lenskit.knn.item.ItemItemScorer;
import org.grouplens.lenskit.knn.item.model.ItemItemBuildContext;
import org.grouplens.lenskit.knn.item.model.ItemItemBuildContextProvider;
import org.grouplens.lenskit.scored.ScoredId;
import org.grouplens.lenskit.transform.normalize.DefaultUserVectorNormalizer;
import org.grouplens.lenskit.vectors.SparseVector;
import org.hamcrest.Matchers;

import java.io.File;
import java.util.List;

public class Runner {
	public static void main(String args[]) throws Exception {
		EventFormat eventFormat = new DelimitedColumnEventFormat(new RatingEventType());
		LenskitConfiguration config = new LenskitConfiguration();
		config.bind(ItemScorer.class).to(ItemItemScorer.class);
		config.bind(BaselineScorer.class, ItemScorer.class).to(UserMeanItemScorer.class);
		config.bind(UserMeanBaseline.class, ItemScorer.class)
				.to(ItemMeanRatingItemScorer.class);
		/*config.bind(UserVectorNormalizer.class)
				.to(BaselineSubtractingUserVectorNormalizer.class);*/


		SimpleEvaluator evaluator = new SimpleEvaluator();

		evaluator.addMetric(new RMSEPredictMetric());

		NDCGPredictMetric ndcg = new NDCGPredictMetric();
		evaluator.addMetric(ndcg);

		NDCGTopNMetric ndcgMetric = new NDCGTopNMetric("0", "1", 3, ItemSelectors.allItems(), ItemSelectors.trainingItems());
		evaluator.addMetric(ndcgMetric);

		MAPTopNMetric map = new MAPTopNMetric("0", "1", 3, ItemSelectors.allItems(), ItemSelectors.trainingItems(), ItemSelectors.testRatingMatches(Matchers.greaterThan(2.5)));
		evaluator.addMetric(map);

		ExampleMetric serendipityMetric = new ExampleMetric();
		evaluator.addMetric(serendipityMetric);

		evaluator.addAlgorithm(new AlgorithmInstance("itemitem", config));
		//DataSource dataSource = new GenericDataSource("fileName10", new TextEventDAO(new File("D:\\bigdata\\movielens\\ml-100k\\u.data"), eventFormat));
		DataSource dataSource = new GenericDataSource("split", new TextEventDAO(new File("D:\\bigdata\\movielens\\fake\\all_ratings_extended"), eventFormat));
		//DataSource dataSource2 = new GenericDataSource("file2", new TextEventDAO(new File("D:\\bigdata\\movielens\\fake\\test"), eventFormat));

		evaluator.setOutputPath("out.csv");
		evaluator.setUserOutputPath("user.csv");
		evaluator.setPredictOutputPath("item.csv");

		CrossfoldTask task = new CrossfoldTask("task");
		task.setHoldout(3);
		task.setPartitions(2);
		task.setSource(dataSource);
		evaluator.addDataset(task);
		//evaluator.addDataset(dataSource, 2);
		//evaluator.addDataset(dataSource, dataSource2);
		//evaluator.call();

		/*SimpleEvaluator evaluator1 = new SimpleEvaluator();
		evaluator.addAlgorithm(new AlgorithmInstance("item", config));*/
		EventFormat eventFormat2 = DelimitedColumnEventFormat.create(new RatingEventType());
		//LenskitConfiguration config1 = new LenskitConfiguration();
		//config.bind(EventDAO.class).to(new TextEventDAO(new File("task-crossfold\\train.0.csv"), eventFormat2));
		//TextEventDAO eventDAO =new TextEventDAO(new File("task-crossfold\\train.0.csv"), eventFormat);
		config.bind(EventDAO.class).to(new SimpleFileRatingDAO(new File("task-crossfold\\train.0.csv"), ","));
		LenskitRecommender rec = LenskitRecommender.build(config);
		ItemRecommender irec = rec.getItemRecommender();
		List<ScoredId> list = irec.recommend(7l);
		System.out.println(list);


		EventDAO eventDAO = new SimpleFileRatingDAO(new File("task-crossfold\\train.0.csv"), ",");
		RatingSnapshotDAO.Builder builder = new RatingSnapshotDAO.Builder(eventDAO, false);
		ItemItemBuildContextProvider provider = new ItemItemBuildContextProvider(builder.get(), new DefaultUserVectorNormalizer(), new RatingVectorUserHistorySummarizer());
		ItemItemBuildContext context = provider.get();
		System.out.println(context.itemVector(101l));
		System.out.println(context.getUserItems(7l));
		SparseVector vector = context.itemVector(101l);
		System.out.println(vector.keySet());
	}
}
