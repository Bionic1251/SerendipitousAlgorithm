package mf.lu;

import it.unimi.dsi.fastutil.longs.LongSet;
import it.unimi.dsi.fastutil.longs.LongSortedSet;
import mikera.vectorz.AVector;
import org.grouplens.lenskit.ItemScorer;
import org.grouplens.lenskit.baseline.BaselineScorer;
import org.grouplens.lenskit.basic.AbstractItemScorer;
import org.grouplens.lenskit.collections.LongUtils;
import org.grouplens.lenskit.data.dao.UserEventDAO;
import org.grouplens.lenskit.data.event.Rating;
import org.grouplens.lenskit.data.event.Ratings;
import org.grouplens.lenskit.data.history.History;
import org.grouplens.lenskit.data.history.UserHistory;
import org.grouplens.lenskit.data.pref.PreferenceDomain;
import org.grouplens.lenskit.mf.svd.BiasedMFKernel;
import org.grouplens.lenskit.mf.svd.DomainClampingKernel;
import org.grouplens.lenskit.mf.svd.DotProductKernel;
import org.grouplens.lenskit.vectors.MutableSparseVector;
import org.grouplens.lenskit.vectors.SparseVector;
import org.grouplens.lenskit.vectors.VectorEntry;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import javax.inject.Inject;

public class LuSVDItemScorer extends AbstractItemScorer {
	protected final LuSVDModel model;


	@Inject
	public LuSVDItemScorer(LuSVDModel model) {
		this.model = model;
	}

	@Override
	public void score(long user, @Nonnull MutableSparseVector scores) {
		for (VectorEntry e : scores.view(VectorEntry.State.EITHER)) {
			AVector userVector = model.getUserVector(user);
			AVector itemVector = model.getItemVector(e.getKey());
			if (itemVector == null) {
				scores.unset(e.getKey());
				continue;
			}
			if (userVector == null) {
				userVector = model.getAverageUserVector();
			}
			double score = userVector.dotProduct(itemVector);
			scores.set(e, score);
		}
	}
}
