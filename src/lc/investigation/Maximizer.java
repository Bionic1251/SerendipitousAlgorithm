package lc.investigation;

public class Maximizer extends Optimizer {

	public Maximizer(double serR, double serD, double serU, double unserR, double unserD, double unserU) {
		double diffR = serR - unserR;
		double diffD = serD - unserD;
		double diffU = serU - unserU;
		double sum = diffR + diffD + diffU;
		r = diffR / sum;
		d = diffD / sum;
		u = diffU / sum;
	}

	@Override
	protected void updateWeights() {
		double pwr = 0, pwd = 0, pwu = 0;
		double step = ALPHA;
		if (wr + wd + wu > 1) {
			pwr -= step;
			pwd -= step;
			pwu -= step;
		}
		if (wr + wd + wu < 1) {
			pwr += step;
			pwd += step;
			pwu += step;
		}
		if (wr < 0) {
			pwr += step;
		}
		if (wd < 0) {
			pwd += step;
		}
		if (wu < 0) {
			pwu += step;
		}

		wr += LEARNING_RATE * (r + pwr - REG_TERM * wr);
		wd += LEARNING_RATE * (d + pwd - REG_TERM * wd);
		wu += LEARNING_RATE * (u + pwu - REG_TERM * wu);
	}
}
