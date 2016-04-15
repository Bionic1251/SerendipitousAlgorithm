import evaluationMetric.*;
import it.unimi.dsi.fastutil.longs.LongSet;
import org.grouplens.lenskit.ItemRecommender;
import org.grouplens.lenskit.core.LenskitConfiguration;
import org.grouplens.lenskit.core.LenskitRecommender;
import org.grouplens.lenskit.cursors.Cursor;
import org.grouplens.lenskit.data.dao.EventDAO;
import org.grouplens.lenskit.data.dao.ItemEventDAO;
import org.grouplens.lenskit.data.dao.SimpleFileRatingDAO;
import org.grouplens.lenskit.data.event.Event;
import org.grouplens.lenskit.data.history.ItemEventCollection;
import org.grouplens.lenskit.data.source.DataSource;
import org.grouplens.lenskit.data.source.GenericDataSource;
import org.grouplens.lenskit.data.text.DelimitedColumnEventFormat;
import org.grouplens.lenskit.data.text.RatingEventType;
import org.grouplens.lenskit.data.text.TextEventDAO;
import org.grouplens.lenskit.eval.data.crossfold.CrossfoldTask;
import org.grouplens.lenskit.eval.metrics.topn.ItemSelector;
import org.grouplens.lenskit.eval.metrics.topn.ItemSelectors;
import org.grouplens.lenskit.eval.traintest.SimpleEvaluator;
import org.grouplens.lenskit.scored.ScoredId;
import org.grouplens.lenskit.util.ScoredItemAccumulator;
import org.grouplens.lenskit.util.TopNScoredItemAccumulator;
import org.hamcrest.Matchers;
import util.AlgorithmUtil;
import util.ContentAverageDissimilarity;
import util.Settings;
import util.Util;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.List;
import java.util.Map;
import java.util.Properties;

public class InvestigationRunner {
	public static void main(String[] args) {
		Util.setParameters();
		ContentAverageDissimilarity.create(Settings.DATASET_CONTENT);
		getRecs(AlgorithmUtil.getPersInvestigation());
		getRecs(AlgorithmUtil.getNonPersInvestigation());
	}

	private static List<ScoredId> getRecs(LenskitConfiguration configuration) {
		try {
			configuration.bind(EventDAO.class).to(new SimpleFileRatingDAO(new File(Settings.DATASET), "\t"));
			LenskitRecommender pop = LenskitRecommender.build(configuration);
			ItemRecommender itemRecommender = pop.getItemRecommender();
			List<ScoredId> recs = itemRecommender.recommend(100000l, 10);
			return recs;
		} catch (Exception e) {
			e.printStackTrace();
		}
		return null;
	}
}
