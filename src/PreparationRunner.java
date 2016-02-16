import util.PrepareUtil;

public class PreparationRunner {
	public static void main(String args[]){
		//PrepareUtil.prepareSmallDataset("D:\\bigdata\\movielens\\ml-100k\\u.item");
		//PrepareUtil.prepareBigDataset("D:\\bigdata\\movielens\\hetrec\\movie_genres.dat");
		//PrepareUtil.prepareYahooDataset("D:\\bigdata\\Yahoo Movies\\ratings.txt");
		//PrepareUtil.prepareContentYahooDataset("D:\\bigdata\\Yahoo Movies\\movie_db_yoda");
		PrepareUtil.printStatistics("ml/big/ratings.dat");
	}
}
