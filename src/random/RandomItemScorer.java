package random;

import it.unimi.dsi.fastutil.longs.LongSortedSet;
import org.grouplens.lenskit.basic.AbstractItemScorer;
import org.grouplens.lenskit.vectors.MutableSparseVector;
import org.grouplens.lenskit.vectors.VectorEntry;

import javax.annotation.Nonnull;

public class RandomItemScorer extends AbstractItemScorer {
	@Override
	public void score(long user, @Nonnull MutableSparseVector scores) {
		for (VectorEntry e : scores.view(VectorEntry.State.EITHER)) {
			scores.set(e.getKey(), Math.random());
		}
	}
}
