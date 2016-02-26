import util.PrepareUtil;

public class PreparationRunner {
	public static void main(String args[]){
		//PrepareUtil.prepareSmallDataset("D:\\bigdata\\movielens\\ml-100k\\u.item");
		//PrepareUtil.prepareBigDataset("D:\\bigdata\\movielens\\hetrec\\movie_genres.dat");
		//PrepareUtil.prepareYahooDataset("D:\\bigdata\\Yahoo Movies\\ratings.txt");
		//PrepareUtil.prepareContentYahooDataset("D:\\bigdata\\Yahoo Movies\\movie_db_yoda");
		//PrepareUtil.printUserItemRatingNumber("ml/big/ratings.dat");
		//PrepareUtil.printDissimilarityRating("dataset/ml/small/ratings.dat", "dataset/ml/small/content.dat");
		//PrepareUtil.printUnpopularityRating("dataset/ml/small/ratings.dat");
		PrepareUtil.generateUnpopDataset("dataset/ml/small/ratings.dat", 22);
	}
}
