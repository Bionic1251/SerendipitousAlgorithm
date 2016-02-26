package evaluationMetric;

import it.unimi.dsi.fastutil.longs.LongSet;
import it.unimi.dsi.fastutil.longs.LongSortedSet;
import org.grouplens.lenskit.Recommender;
import org.grouplens.lenskit.data.dao.packed.RatingSnapshotDAO;
import org.grouplens.lenskit.data.history.RatingVectorUserHistorySummarizer;
import org.grouplens.lenskit.eval.Attributed;
import org.grouplens.lenskit.eval.data.traintest.TTDataSet;
import org.grouplens.lenskit.eval.metrics.AbstractMetric;
import org.grouplens.lenskit.eval.metrics.ResultColumn;
import org.grouplens.lenskit.eval.metrics.topn.ItemSelector;
import org.grouplens.lenskit.eval.traintest.TestUser;
import org.grouplens.lenskit.knn.item.model.ItemItemBuildContext;
import org.grouplens.lenskit.knn.item.model.ItemItemBuildContextProvider;
import org.grouplens.lenskit.scored.ScoredId;
import org.grouplens.lenskit.transform.normalize.DefaultUserVectorNormalizer;
import org.grouplens.lenskit.util.statistics.MeanAccumulator;
import org.grouplens.lenskit.vectors.SparseVector;
import util.ContentAverageDissimilarity;
import util.ContentUtil;

import javax.annotation.Nullable;
import java.util.*;

public class ComponentMetric extends AbstractMetric<MeanAccumulator, ComponentMetric.CompRes, ComponentMetric.CompRes> {
	private final int listLen;
	private Map<Long, Double> popMap;
	private MeanAccumulator dAccumulator = new MeanAccumulator();
	private MeanAccumulator uAccumulator = new MeanAccumulator();
	private final ItemSelector candidates;
	private final ItemSelector exclude;
	private final String suffix;

	public ComponentMetric(String suffix, int listLen, ItemSelector candidates, ItemSelector exclude) {
		super(ComponentMetric.CompRes.class, ComponentMetric.CompRes.class);
		this.listLen = listLen;
		this.candidates = candidates;
		this.exclude = exclude;
		this.suffix = suffix;
	}

	@Override
	protected String getSuffix() {
		return suffix;
	}

	@Override
	protected ComponentMetric.CompRes doMeasureUser(TestUser user, MeanAccumulator context) {
		List<ScoredId> recommendations = user.getRecommendations(listLen, candidates, exclude);
		if (recommendations == null || recommendations.isEmpty()) {
			dAccumulator.add(0);
			uAccumulator.add(0);
			return null;
		}
		double dissimilarity = 0;
		if (user.getTrainHistory() == null || user.getTrainHistory().itemSet().isEmpty()) {
			dAccumulator.add(0);
		} else {
			LongSet profile = user.getTrainHistory().itemSet();
			dissimilarity = getDissim(recommendations, profile);
			dAccumulator.add(dissimilarity);
		}
		double unpop = getUnpop(recommendations);
		uAccumulator.add(unpop);
		return new ComponentMetric.CompRes(unpop, dissimilarity);
	}

	private double getUnpop(List<ScoredId> recommendations) {
		double unpop = 0;
		for (ScoredId item : recommendations) {
			unpop = 1 - popMap.get(item.getId());
		}
		unpop = unpop / recommendations.size();
		return unpop;
	}

	private double getDissim(List<ScoredId> recommendations, LongSet ratedItems) {
		double dissim = 0;
		for (ScoredId item : recommendations) {
			dissim = getDissim(item.getId(), ratedItems);
		}
		dissim = dissim / (double) recommendations.size();
		return dissim;
	}

	private double getDissim(Long itemId, LongSet ratedIds) {
		double dissim = 0;
		ContentAverageDissimilarity dissimilarity = ContentAverageDissimilarity.getInstance();
		Map<Long, SparseVector> dMap = dissimilarity.getItemContentMap();
		SparseVector item = dMap.get(itemId);
		for (Long ratedItemId : ratedIds) {
			SparseVector ratedItem = dMap.get(ratedItemId);
			dissim += 1.0 - ContentUtil.getCosine(item, ratedItem);
		}
		dissim = dissim / ratedIds.size();
		return dissim;
	}

	@Override
	protected ComponentMetric.CompRes getTypedResults(MeanAccumulator context) {
		return new ComponentMetric.CompRes(uAccumulator.getMean(), dAccumulator.getMean());
	}

	@Nullable
	@Override
	public MeanAccumulator createContext(Attributed algorithm, TTDataSet dataSet, Recommender recommender) {
		updateExpectedItems(dataSet);
		return new MeanAccumulator();
	}

	private void updateExpectedItems(TTDataSet dataSet) {
		RatingSnapshotDAO.Builder builder = new RatingSnapshotDAO.Builder(dataSet.getTrainingDAO(), false);
		ItemItemBuildContextProvider provider = new ItemItemBuildContextProvider(builder.get(), new DefaultUserVectorNormalizer(), new RatingVectorUserHistorySummarizer());
		updatePopMap(provider.get());
	}

	private void updatePopMap(ItemItemBuildContext dataContext) {
		popMap = new HashMap<Long, Double>();
		Set<Long> userSet = new HashSet<Long>();
		List<Container<Integer>> popItemContainers = new ArrayList<Container<Integer>>();
		LongSortedSet itemSet = dataContext.getItems();
		SparseVector itemVector;
		int maxVal = 0;
		for (Long itemId : itemSet) {
			itemVector = dataContext.itemVector(itemId);
			maxVal = Math.max(maxVal, itemVector.values().size());
			popItemContainers.add(new Container(itemId, itemVector.values().size()));
			userSet.addAll(itemVector.keySet());
		}
		for (Container<Integer> container : popItemContainers) {
			popMap.put(container.getId(), (double) container.getValue() / (double) maxVal);
		}
	}

	public static class CompRes {
		@ResultColumn("Unpop")
		public final double unpop;
		@ResultColumn("Dissim")
		public final double dissim;

		public CompRes(double unpop, double dissim) {
			this.unpop = unpop;
			this.dissim = dissim;
		}
	}
}
