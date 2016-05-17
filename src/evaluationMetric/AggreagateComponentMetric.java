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
import util.PrepareUtil;
import util.Settings;

import javax.annotation.Nullable;
import java.util.*;

public class AggreagateComponentMetric extends AbstractMetric<MeanAccumulator, AggreagateComponentMetric.CompRes, AggreagateComponentMetric.CompRes> {
	private Map<Long, Double> popMap;
	private MeanAccumulator dAccumulator1 = new MeanAccumulator();
	private MeanAccumulator dAccumulator5 = new MeanAccumulator();
	private MeanAccumulator dAccumulator10 = new MeanAccumulator();
	private MeanAccumulator dAccumulator15 = new MeanAccumulator();
	private MeanAccumulator dAccumulator20 = new MeanAccumulator();
	private MeanAccumulator uAccumulator1 = new MeanAccumulator();
	private MeanAccumulator uAccumulator5 = new MeanAccumulator();
	private MeanAccumulator uAccumulator10 = new MeanAccumulator();
	private MeanAccumulator uAccumulator15 = new MeanAccumulator();
	private MeanAccumulator uAccumulator20 = new MeanAccumulator();
	private final ItemSelector candidates;
	private final ItemSelector exclude;
	private final String suffix;
	private int count;

	public AggreagateComponentMetric(String suffix, ItemSelector candidates, ItemSelector exclude) {
		super(AggreagateComponentMetric.CompRes.class, AggreagateComponentMetric.CompRes.class);
		this.candidates = candidates;
		this.exclude = exclude;
		this.suffix = suffix;
	}

	@Override
	protected String getSuffix() {
		return suffix;
	}

	@Override
	protected AggreagateComponentMetric.CompRes doMeasureUser(TestUser user, MeanAccumulator context) {
		count++;
		if (count % 100 == 0) {
			System.out.println(count + " users " + this.getClass());
		}
		List<ScoredId> recommendations = user.getRecommendations(20, candidates, exclude);
		if (recommendations == null || recommendations.isEmpty()) {
			dAccumulator1.add(0);
			dAccumulator5.add(0);
			dAccumulator10.add(0);
			dAccumulator15.add(0);
			dAccumulator20.add(0);
			uAccumulator1.add(0);
			uAccumulator5.add(0);
			uAccumulator10.add(0);
			uAccumulator15.add(0);
			uAccumulator20.add(0);
			return null;
		}
		measure(recommendations, user, 1, dAccumulator1, uAccumulator1);
		measure(recommendations, user, 5, dAccumulator5, uAccumulator5);
		measure(recommendations, user, 10, dAccumulator10, uAccumulator10);
		measure(recommendations, user, 15, dAccumulator15, uAccumulator15);
		measure(recommendations, user, 20, dAccumulator20, uAccumulator20);
		return null;
	}

	private void measure(List<ScoredId> recs, TestUser user, int listLen, MeanAccumulator dAccumulator,
						 MeanAccumulator uAccumulator) {
		List<ScoredId> recommendations = recs;
		if (recs.size() > listLen) {
			recommendations = recs.subList(0, listLen);
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
	}

	private double getUnpop(List<ScoredId> recommendations) {
		double unpop = 0;
		for (ScoredId item : recommendations) {
			if (!popMap.containsKey(item.getId())) {
				unpop += 1.0;
			} else {
				unpop += 1 - popMap.get(item.getId());
			}
		}
		unpop = unpop / recommendations.size();
		return unpop;
	}

	private double getDissim(List<ScoredId> recommendations, LongSet ratedItems) {
		double dissim = 0;
		for (ScoredId item : recommendations) {
			dissim += getDissim(item.getId(), ratedItems);
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
			dissim += 1.0 - ContentUtil.getJaccard(item, ratedItem);
		}
		dissim = dissim / ratedIds.size();
		return dissim;
	}

	@Override
	protected AggreagateComponentMetric.CompRes getTypedResults(MeanAccumulator context) {
		return new AggreagateComponentMetric.CompRes(uAccumulator1.getMean(), uAccumulator5.getMean(), uAccumulator10.getMean(), uAccumulator15.getMean(), uAccumulator20.getMean(),
				dAccumulator1.getMean(), dAccumulator5.getMean(), dAccumulator10.getMean(), dAccumulator15.getMean(), dAccumulator20.getMean());
	}

	@Nullable
	@Override
	public MeanAccumulator createContext(Attributed algorithm, TTDataSet dataSet, Recommender recommender) {
		System.out.println("createContext " + this.getClass());
		updateExpectedItems(dataSet);
		count = 0;
		updateAccumulators();
		return new MeanAccumulator();
	}

	private void updateAccumulators(){
		uAccumulator1 = new MeanAccumulator();
		uAccumulator5 = new MeanAccumulator();
		uAccumulator10 = new MeanAccumulator();
		uAccumulator15 = new MeanAccumulator();
		uAccumulator20 = new MeanAccumulator();
		dAccumulator1 = new MeanAccumulator();
		dAccumulator5 = new MeanAccumulator();
		dAccumulator10 = new MeanAccumulator();
		dAccumulator15 = new MeanAccumulator();
		dAccumulator20 = new MeanAccumulator();
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

	/*private void updateExpectedItems(TTDataSet dataSet) {
		popMap = PrepareUtil.getNormalizedPopMap(dataSet.getTrainingData().getName(), "\t");
	}*/

	public static class CompRes {
		@ResultColumn("Unpop1")
		public final double unpop1;
		@ResultColumn("Unpop5")
		public final double unpop5;
		@ResultColumn("Unpop10")
		public final double unpop10;
		@ResultColumn("Unpop15")
		public final double unpop15;
		@ResultColumn("Unpop20")
		public final double unpop20;
		@ResultColumn("Dissim1")
		public final double dissim1;
		@ResultColumn("Dissim5")
		public final double dissim5;
		@ResultColumn("Dissim10")
		public final double dissim10;
		@ResultColumn("Dissim15")
		public final double dissim15;
		@ResultColumn("Dissim20")
		public final double dissim20;

		public CompRes(double unpop1, double unpop5, double unpop10, double unpop15, double unpop20, double dissim1, double dissim5, double dissim10, double dissim15, double dissim20) {
			this.unpop1 = unpop1;
			this.unpop5 = unpop5;
			this.unpop10 = unpop10;
			this.unpop15 = unpop15;
			this.unpop20 = unpop20;
			this.dissim1 = dissim1;
			this.dissim5 = dissim5;
			this.dissim10 = dissim10;
			this.dissim15 = dissim15;
			this.dissim20 = dissim20;
		}
	}
}
