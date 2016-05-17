package evaluationMetric;

import org.grouplens.lenskit.Recommender;
import org.grouplens.lenskit.eval.Attributed;
import org.grouplens.lenskit.eval.data.traintest.TTDataSet;
import org.grouplens.lenskit.eval.metrics.AbstractMetric;
import org.grouplens.lenskit.eval.metrics.topn.ItemSelector;
import org.grouplens.lenskit.eval.metrics.topn.ItemSelectors;
import org.grouplens.lenskit.eval.traintest.TestUser;
import org.grouplens.lenskit.scored.ScoredId;

import javax.annotation.Nullable;
import java.io.PrintWriter;
import java.util.List;

public class WriterMetric extends AbstractMetric {
	PrintWriter printWriter;

	public WriterMetric() {
		super(Object.class, Object.class);
	}

	@Override
	protected Object doMeasureUser(TestUser user, Object context) {
		List<ScoredId> list = user.getRecommendations(Integer.MAX_VALUE, ItemSelectors.allItems(), ItemSelectors.trainingItems());
		printWriter.print(user.getUserId());
		for (ScoredId scoredId : list) {
			printWriter.print("\t" + scoredId.getId());
		}
		printWriter.println();
		return null;
	}

	@Override
	protected Object getTypedResults(Object context) {
		return null;
	}

	@Nullable
	@Override
	public Object createContext(Attributed algorithm, TTDataSet dataSet, Recommender recommender) {
		try {
			if (printWriter != null) {
				printWriter.close();
			}
			String[] split = dataSet.getName().split("\\.");
			String fileName = split[split.length - 1];
			printWriter = new PrintWriter(fileName);
		} catch (Exception e) {
			e.printStackTrace();
		}
		return null;
	}
}
