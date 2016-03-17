package util;

import lc.Normalizer;

public class AverageAggregate {
	private final UserStatistics r;
	private final UserStatistics d;
	private final UserStatistics u;

	public AverageAggregate(UserStatistics r, UserStatistics d, UserStatistics u) {
		this.r = r;
		this.d = d;
		this.u = u;
	}

	public UserStatistics getR() {
		return r;
	}

	public UserStatistics getD() {
		return d;
	}

	public UserStatistics getU() {
		return u;
	}
}
