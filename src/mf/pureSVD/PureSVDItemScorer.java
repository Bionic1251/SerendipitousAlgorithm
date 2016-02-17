package mf.pureSVD;

import mikera.vectorz.AVector;
import org.grouplens.lenskit.basic.AbstractItemScorer;
import org.grouplens.lenskit.vectors.MutableSparseVector;
import org.grouplens.lenskit.vectors.VectorEntry;

import javax.annotation.Nonnull;
import javax.inject.Inject;

public class PureSVDItemScorer extends AbstractItemScorer {
	protected final PureSVDModel model;

	@Inject
	public PureSVDItemScorer(PureSVDModel model) {
		this.model = model;
	}

	@Override
	public void score(long user, @Nonnull MutableSparseVector scores) {
		for (VectorEntry e : scores.view(VectorEntry.State.EITHER)) {
			AVector userVector = model.getUserVector(user);
			AVector itemVector = model.getItemVector(e.getKey());
			if (itemVector == null) {
				itemVector = model.getAverageUserVector();
			}
			if (userVector == null) {
				userVector = model.getAverageUserVector();
			}
			double score = userVector.dotProduct(itemVector);
			scores.set(e, score);
		}
	}
}
