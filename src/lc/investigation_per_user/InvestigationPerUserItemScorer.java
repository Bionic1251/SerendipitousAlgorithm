package lc.investigation_per_user;

import org.grouplens.lenskit.basic.AbstractItemScorer;
import org.grouplens.lenskit.vectors.MutableSparseVector;
import org.grouplens.lenskit.vectors.VectorEntry;

import javax.annotation.Nonnull;
import javax.inject.Inject;

public class InvestigationPerUserItemScorer extends AbstractItemScorer {

	@Inject
	public InvestigationPerUserItemScorer(InvestigationPerUserModel lcModel) {
	}

	@Override
	public void score(long user, @Nonnull MutableSparseVector scores) {
		for (VectorEntry e : scores.view(VectorEntry.State.EITHER)) {
			scores.unset(e);
		}
	}
}
