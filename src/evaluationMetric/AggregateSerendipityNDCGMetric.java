/*
 * LensKit, an open source recommender systems toolkit.
 * Copyright 2010-2014 LensKit Contributors.  See CONTRIBUTORS.md.
 * Work on LensKit has been funded by the National Science Foundation under
 * grants IIS 05-34939, 08-08692, 08-12148, and 10-17697.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation; either version 2.1 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this program; if not, write to the Free Software Foundation, Inc., 51
 * Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 */
package evaluationMetric;

import it.unimi.dsi.fastutil.longs.LongArrayList;
import it.unimi.dsi.fastutil.longs.LongList;
import it.unimi.dsi.fastutil.longs.LongSortedSet;
import org.grouplens.lenskit.Recommender;
import org.grouplens.lenskit.data.dao.UserEventDAO;
import org.grouplens.lenskit.data.dao.packed.RatingSnapshotDAO;
import org.grouplens.lenskit.data.event.Event;
import org.grouplens.lenskit.data.history.RatingVectorUserHistorySummarizer;
import org.grouplens.lenskit.data.history.UserHistory;
import org.grouplens.lenskit.data.pref.PreferenceDomain;
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
import org.grouplens.lenskit.vectors.MutableSparseVector;
import org.grouplens.lenskit.vectors.SparseVector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import util.ContentAverageDissimilarity;
import util.ContentUtil;

import java.io.File;
import java.io.PrintWriter;
import java.util.*;

import static java.lang.Math.log;

/**
 * @author <a href="http://www.grouplens.org">GroupLens Research</a>
 */
public class AggregateSerendipityNDCGMetric extends AbstractMetric<MeanAccumulator, AggregateSerendipityNDCGMetric.AggregateResult, AggregateSerendipityNDCGMetric.AggregateResult> {
	private static final Logger logger = LoggerFactory.getLogger(AggregateSerendipityNDCGMetric.class);

	private final ItemSelector candidates;
	private final ItemSelector exclude;
	private final String prefix;
	private final String suffix;
	private SparseVector unpopularityVector;
	private final Map<Long, SparseVector> itemContentMap;
	private final double relevanceTheshold;
	private final double unpopTheshold;
	private final double dissimTheshold;
	private final PreferenceDomain domain;

	private MeanAccumulator context1;
	private MeanAccumulator context5;
	private MeanAccumulator context10;
	private MeanAccumulator context15;
	private MeanAccumulator context20;
	private MeanAccumulator context25;
	private MeanAccumulator context30;

	/**
	 * Construct a new nDCG Top-N metric.
	 *
	 * @param pre        the prefix label for this evaluation, or {@code null} for no prefix.
	 * @param sfx        the suffix label for this evaluation, or {@code null} for no suffix.
	 * @param candidates The candidate selector.
	 * @param exclude    The exclude selector.
	 */
	public AggregateSerendipityNDCGMetric(String pre, String sfx, ItemSelector candidates, ItemSelector exclude,
										  double threshold, double unpopThreshold, double dissThreshold, PreferenceDomain domain) {
		super(AggregateResult.class, AggregateResult.class);
		suffix = sfx;
		prefix = pre;
		this.candidates = candidates;
		this.exclude = exclude;
		ContentAverageDissimilarity contentAverageDissimilarity = ContentAverageDissimilarity.getInstance();
		this.itemContentMap = contentAverageDissimilarity.getItemContentMap();
		relevanceTheshold = threshold;
		this.unpopTheshold = unpopThreshold;
		dissimTheshold = dissThreshold;
		this.domain = domain;
	}


	@Override
	protected String getPrefix() {
		return prefix;
	}

	@Override
	protected String getSuffix() {
		return suffix;
	}

	private double getVal(Long itemId, Double rating, TestUser user) {
		if (rating <= relevanceTheshold) {
			return 0;
		}
		if (!unpopularityVector.containsKey(itemId)) {
			return 0;
		}
		if (user.getTrainHistory() == null || user.getTrainHistory().isEmpty()) {
			return unpopularityVector.get(itemId);
		}
		double unpop = unpopularityVector.get(itemId);
		if (unpop < unpopTheshold) {
			return 0.0;
		}
		List<Long> itemIdList = new ArrayList<Long>();
		Collection<Event> events = user.getTrainHistory();
		for (Event event : events) {
			itemIdList.add(event.getItemId());
		}
		double dissimilarity = getAverageDissimilarity(itemIdList, itemId);
		if (dissimilarity < dissimTheshold) {
			return 0.0;
		}
		double value = dissimilarity + unpop + rating / domain.getMaximum();
		return value;
	}

	private double computeDCG(List<Long> items, SparseVector values, TestUser user) {
		final double lg2 = log(2);

		double gain = 0;
		int rank = 0;

		Iterator<Long> iit = items.iterator();
		while (iit.hasNext()) {
			final Long item = iit.next();
			double v = getVal(item, values.get(item, 0), user);
			rank++;
			if (rank < 2) {
				gain += v;
			} else {
				gain += v * lg2 / log(rank);
			}
		}

		return gain;
	}

	@Override
	public AggregateResult doMeasureUser(TestUser user, MeanAccumulator context) {
		List<ScoredId> recommendations = user.getRecommendations(30, candidates, exclude);
		if (recommendations == null) {
			return null;
		}
		double ndcg1 = measureUser(user, context1, recommendations, 1);
		double ndcg5 = measureUser(user, context5, recommendations, 5);
		double ndcg10 = measureUser(user, context10, recommendations, 10);
		double ndcg15 = measureUser(user, context15, recommendations, 15);
		double ndcg20 = measureUser(user, context20, recommendations, 20);
		double ndcg25 = measureUser(user, context25, recommendations, 25);
		double ndcg30 = measureUser(user, context30, recommendations, 30);
		return new AggregateResult(ndcg1, ndcg5, ndcg10, ndcg15, ndcg20, ndcg25, ndcg30);
	}

	public double measureUser(TestUser user, MeanAccumulator context, List<ScoredId> recommendations, int listSize) {
		if (recommendations.size() > listSize) {
			recommendations = new ArrayList<ScoredId>(recommendations.subList(0, listSize));
		}
		SparseVector ratings = user.getTestRatings();
		List<Long> ideal = getSortedList(ratings, user);
		if (ideal.size() > listSize) {
			ideal = ideal.subList(0, listSize);
		}
		double idealGain = computeDCG(ideal, ratings, user);
		if (idealGain == 0.0) {
			return 0.0;
		}

		LongList actual = new LongArrayList(recommendations.size());
		for (ScoredId id : recommendations) {
			actual.add(id.getId());
		}
		double gain = computeDCG(actual, ratings, user);

		double score = gain / idealGain;

		context.add(score);
		return score;
	}

	private List<Long> getSortedList(SparseVector ratings, TestUser user) {
		List<Container<Double>> containerList = new ArrayList<Container<Double>>();
		for (long key : ratings.keySet()) {
			containerList.add(new Container<Double>(key, getVal(key, ratings.get(key), user)));
		}
		Collections.sort(containerList);
		Collections.reverse(containerList);
		List<Long> ids = new ArrayList<Long>();
		for (Container<Double> container : containerList) {
			ids.add(container.getId());
		}
		return ids;
	}

	@Override
	public MeanAccumulator createContext(Attributed algo, TTDataSet ds, Recommender rec) {
		context1 = new MeanAccumulator();
		context5 = new MeanAccumulator();
		context10 = new MeanAccumulator();
		context15 = new MeanAccumulator();
		context20 = new MeanAccumulator();
		context25 = new MeanAccumulator();
		context30 = new MeanAccumulator();
		updateExpectedItems(ds);
		return new MeanAccumulator();
	}

	private void updateExpectedItems(TTDataSet dataSet) {
		RatingSnapshotDAO.Builder builder = new RatingSnapshotDAO.Builder(dataSet.getTrainingDAO(), false);
		ItemItemBuildContextProvider itemItemProvider = new ItemItemBuildContextProvider(builder.get(), new DefaultUserVectorNormalizer(), new RatingVectorUserHistorySummarizer());
		updateSurpriseMap(itemItemProvider.get());
	}

	private void updateSurpriseMap(ItemItemBuildContext dataContext) {
		updateUnpopularity(dataContext);
	}

	/*private void updateDissimilarity(ItemItemBuildContext dataContext) {
		System.out.println("Calculating dissimilarity for the serendipity evaluation metric");
		int i = 0;
		for (Map.Entry<Long, SparseVector> userEntry : surpriseMap.entrySet()) {
			i++;
			if (i % 100 == 0) {
				System.out.println(i + " users processed");
			}
			MutableSparseVector vector = (MutableSparseVector) userEntry.getValue();
			Collection<Long> ratedItems = dataContext.getUserItems(userEntry.getKey());
			for (Long itemId : vector.keySet()) {
				double value = vector.get(itemId) + getAverageDissimilarity(ratedItems, itemId);
				vector.set(itemId, value);
			}
		}
	}*/

	private double getAverageDissimilarity(Collection<Long> ratedItems, Long itemId) {
		double avgSim = 0.0;
		for (Long ratedItemId : ratedItems) {
			avgSim += 1 - ContentUtil.getCosine(itemContentMap.get(ratedItemId), itemContentMap.get(itemId));
		}
		return avgSim / ratedItems.size();
	}

	private void updateUnpopularity(ItemItemBuildContext dataContext) {
		Set<Long> userSet = new HashSet<Long>();
		LongSortedSet itemSet = dataContext.getItems();
		SparseVector itemVector;
		MutableSparseVector userVector = MutableSparseVector.create(itemSet);
		double maxPop = 0;
		for (Long itemId : itemSet) {
			itemVector = dataContext.itemVector(itemId);
			userVector.set(itemId, (double) itemVector.values().size());
			maxPop = Math.max(maxPop, itemVector.values().size());
			userSet.addAll(itemVector.keySet());
		}
		for (Long id : userVector.keySet()) {
			double normVal = 1 - userVector.get(id) / maxPop;
			userVector.set(id, normVal);
		}
		unpopularityVector = userVector;
	}

	@Override
	protected AggregateResult getTypedResults(MeanAccumulator context) {
		return new AggregateResult(context1.getMean(), context5.getMean(), context10.getMean(), context15.getMean(), context20.getMean(), context25.getMean(), context30.getMean());
	}

	public static class AggregateResult {
		@ResultColumn("nDCG1")
		public final double nDCG1;

		@ResultColumn("nDCG5")
		public final double nDCG5;

		@ResultColumn("nDCG10")
		public final double nDCG10;

		@ResultColumn("nDCG15")
		public final double nDCG15;

		@ResultColumn("nDCG20")
		public final double nDCG20;

		@ResultColumn("nDCG25")
		public final double nDCG25;

		@ResultColumn("nDCG30")
		public final double nDCG30;

		public AggregateResult(double nDCG1, double nDCG5, double nDCG10, double nDCG15, double nDCG20, double nDCG25, double nDCG30) {
			this.nDCG1 = nDCG1;
			this.nDCG5 = nDCG5;
			this.nDCG10 = nDCG10;
			this.nDCG15 = nDCG15;
			this.nDCG20 = nDCG20;
			this.nDCG25 = nDCG25;
			this.nDCG30 = nDCG30;
		}
	}

}
