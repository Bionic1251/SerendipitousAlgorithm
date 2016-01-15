import evaluationMetric.SMetric;
import org.grouplens.lenskit.ItemScorer;
import org.grouplens.lenskit.baseline.BaselineScorer;
import org.grouplens.lenskit.baseline.ItemMeanRatingItemScorer;
import org.grouplens.lenskit.baseline.UserMeanBaseline;
import org.grouplens.lenskit.baseline.UserMeanItemScorer;
import org.grouplens.lenskit.core.LenskitConfiguration;
import org.grouplens.lenskit.data.dao.EventDAO;
import org.grouplens.lenskit.data.dao.SimpleFileRatingDAO;
import org.grouplens.lenskit.data.dao.packed.RatingSnapshotDAO;
import org.grouplens.lenskit.data.history.RatingVectorUserHistorySummarizer;
import org.grouplens.lenskit.data.source.CSVDataSourceBuilder;
import org.grouplens.lenskit.data.source.DataSource;
import org.grouplens.lenskit.data.source.GenericDataSource;
import org.grouplens.lenskit.data.text.DelimitedColumnEventFormat;
import org.grouplens.lenskit.data.text.EventFormat;
import org.grouplens.lenskit.data.text.RatingEventType;
import org.grouplens.lenskit.data.text.TextEventDAO;
import org.grouplens.lenskit.eval.data.crossfold.CrossfoldTask;
import org.grouplens.lenskit.eval.metrics.topn.ItemSelectors;
import org.grouplens.lenskit.eval.metrics.topn.NDCGTopNMetric;
import org.grouplens.lenskit.eval.traintest.SimpleEvaluator;
import org.grouplens.lenskit.knn.item.ItemItemScorer;
import org.grouplens.lenskit.knn.item.model.ItemItemBuildContext;
import org.grouplens.lenskit.knn.item.model.ItemItemBuildContextProvider;
import org.grouplens.lenskit.transform.normalize.DefaultUserVectorNormalizer;

import java.io.File;

public class EvaluatorRunner {
	private static final int CROSSFOLD_NUMBER = 2;
	private static final int HOLDOUT_NUMBER = 3;
	private static final String DATASET_PATH = "D:\\bigdata\\movielens\\fake\\all_ratings_extended";
	private static final String TRAIN_TEST_FOLDER_NAME = "task";
	private static final String TRAIN_PATH = TRAIN_TEST_FOLDER_NAME + "-crossfold\\train.";
	private static final String TEST_PATH = TRAIN_TEST_FOLDER_NAME + "-crossfold\\test.";

	public static void main(String args[]) {
		splitDataset();
		int i = 0;
		String trainDatasetPath = TRAIN_PATH + i + ".csv";
		String testDatasetPath = TEST_PATH + i + ".csv";
		EventDAO trainData = new SimpleFileRatingDAO(new File(trainDatasetPath), ",");
		RatingSnapshotDAO.Builder builder = new RatingSnapshotDAO.Builder(trainData, false);
		ItemItemBuildContextProvider provider = new ItemItemBuildContextProvider(builder.get(), new DefaultUserVectorNormalizer(), new RatingVectorUserHistorySummarizer());
		ItemItemBuildContext context = provider.get();

		SMetric metric = new SMetric(context);
		SimpleEvaluator evaluator = new SimpleEvaluator();

		evaluator.addMetric(new NDCGTopNMetric("2", "", 2, ItemSelectors.allItems(), ItemSelectors.trainingItems()));
		//evaluator.addMetric(new evaluationMetric.SerendipityTopNMetric(2));
		evaluator.addMetric(metric);

		LenskitConfiguration config = new LenskitConfiguration();
		config.bind(ItemScorer.class).to(ItemItemScorer.class);
		config.bind(BaselineScorer.class, ItemScorer.class).to(UserMeanItemScorer.class);
		config.bind(UserMeanBaseline.class, ItemScorer.class).to(ItemMeanRatingItemScorer.class);

		CSVDataSourceBuilder trainDataSourceBuilder = new CSVDataSourceBuilder(new File(trainDatasetPath));
		CSVDataSourceBuilder testDataSourceBuilder = new CSVDataSourceBuilder(new File(testDatasetPath));
		evaluator.addDataset(trainDataSourceBuilder.build(), testDataSourceBuilder.build());
		evaluator.setOutputPath("out.csv");
		evaluator.setUserOutputPath("user.csv");
		evaluator.setPredictOutputPath("item.csv");
		evaluator.addAlgorithm("itemitemAlg", config);
		try {
			evaluator.call();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	private static void splitDataset() {
		SimpleEvaluator evaluator = new SimpleEvaluator();
		EventFormat eventFormat = new DelimitedColumnEventFormat(new RatingEventType());
		DataSource dataSource = new GenericDataSource("split", new TextEventDAO(new File(DATASET_PATH), eventFormat));
		CrossfoldTask task = new CrossfoldTask(TRAIN_TEST_FOLDER_NAME);
		task.setHoldout(HOLDOUT_NUMBER);
		task.setPartitions(CROSSFOLD_NUMBER);
		task.setSource(dataSource);
		evaluator.addDataset(task);
	}
}
