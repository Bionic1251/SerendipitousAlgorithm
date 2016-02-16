package evaluationMetric;

import it.unimi.dsi.fastutil.longs.LongSortedSet;
import org.grouplens.lenskit.Recommender;
import org.grouplens.lenskit.data.dao.packed.RatingSnapshotDAO;
import org.grouplens.lenskit.data.history.RatingVectorUserHistorySummarizer;
import org.grouplens.lenskit.eval.Attributed;
import org.grouplens.lenskit.eval.data.traintest.TTDataSet;
import org.grouplens.lenskit.eval.metrics.AbstractMetric;
import org.grouplens.lenskit.eval.metrics.ResultColumn;
import org.grouplens.lenskit.eval.metrics.topn.ItemSelector;
import org.grouplens.lenskit.eval.metrics.topn.NDCGTopNMetric;
import org.grouplens.lenskit.eval.metrics.topn.PrecisionRecallTopNMetric;
import org.grouplens.lenskit.eval.traintest.TestUser;
import org.grouplens.lenskit.knn.item.model.ItemItemBuildContext;
import org.grouplens.lenskit.knn.item.model.ItemItemBuildContextProvider;
import org.grouplens.lenskit.transform.normalize.DefaultUserVectorNormalizer;
import org.grouplens.lenskit.util.statistics.MeanAccumulator;
import org.grouplens.lenskit.vectors.SparseVector;

import java.util.*;


public class AggregateMetric extends AbstractMetric<MeanAccumulator, AggregateMetric.Result, AggregateMetric.Result> {
	private PrecisionRecallTopNMetric.Context precisionContext;
	private final PrecisionRecallTopNMetric precisionRecallTopNMetric;
	private MeanAccumulator ndcgContext;
	private final NDCGTopNMetric ndcgTopNMetric;

	public AggregateMetric(String suffix, int listSize, ItemSelector candidates, ItemSelector exclude, ItemSelector goodItems) {
		super(AggregateMetric.Result.class, AggregateMetric.Result.class);
		precisionRecallTopNMetric = new PrecisionRecallTopNMetric("new", suffix, listSize, candidates, exclude, goodItems);
		ndcgTopNMetric = new NDCGTopNMetric("new", suffix, listSize, candidates, exclude);
	}

	@Override
	protected AggregateMetric.Result doMeasureUser(TestUser user, MeanAccumulator context) {
		precisionRecallTopNMetric.doMeasureUser(user, precisionContext);
		ndcgTopNMetric.doMeasureUser(user, ndcgContext);

		return new AggregateMetric.Result(0.0, 1);
	}

	@Override
	protected AggregateMetric.Result getTypedResults(MeanAccumulator context) {
		List<Object> precisionList = precisionRecallTopNMetric.getResults(precisionContext);
		List<Object> ndcgList = ndcgTopNMetric.getResults(ndcgContext);
		double res = ndcgContext.getMean();
		System.out.println(res);
		return new AggregateMetric.Result(context.getMean(), context.getCount());
	}

	@Override
	public MeanAccumulator createContext(Attributed algorithm, TTDataSet dataSet, Recommender recommender) {
		precisionContext = precisionRecallTopNMetric.createContext(algorithm, dataSet, recommender);
		ndcgContext = new MeanAccumulator();
		return new MeanAccumulator();
	}

	public static class Result {
		@ResultColumn("NDCG")
		public final double utility;

		@ResultColumn("precision")
		public final long count;

		public Result(double util, long count) {
			utility = util;
			this.count = count;
		}
	}
}
