public class Example {
	public static void main(String args[]) {
		double log2 = Math.log(2);
		double lExp = log2 / Math.log(3);
		double a1 = 4 + 4 * lExp;
		double b1 = 6 + 2 * lExp;
		double a2 = 6 + 4 * lExp;
		double b2 = 9 + lExp;
		double a3 = 6 + 4 * lExp;
		double b3 = 8 * 2 * lExp;
		double res = (a1 / b1 + a2 / b2 + a3 / b3 + 7) / 10.0;
		System.out.println(res);
		System.out.println((6 + 4 * lExp) / (9 + lExp));
	}
}
