package util;

public class HTMLGenerator {
	public static String header = "<!doctype html>\n" +
			"\n" +
			"<html lang=\"en\">\n" +
			"<head>\n" +
			"  <meta charset=\"utf-8\">\n" +
			"  <title>Experiment</title>\n" +
			"  <style type=\"text/css\">\n" +
			"   p { \n" +
			"    font-size: 150%;\n" +
			"   }\n" +
			"  </style>\n" +
			"</head>\n" +
			"<body>\n" +
			"<h1 align=\"center\">Recommender System Experiment</h1>\n" +
			"<p>Please indicate movies you like, know and expect to be recommended to you. You can look at the description available on <b>IMDb website</b> or watch a trailer on <b>YouTube</b> (the links are provided).<br>\n" +
			"If you have not watched a suggested movie, indicate whether it seems interesting to you.</p>\n" +
			"<p style=\"color: red\">By filling this form you give your consent to use your data for research purposes including publishing online (your data will be anonymized). If you wish to withdraw from this study, please contact me (deigkotk@student.jyu.fi).</p>" +
			"<form action=\"result.php\" method=\"post\"><table style=\"width:100%\" border=\"1\">\n" +
			"<tr>\n" +
			"\t<th width=\"30%\">\n" +
			"\t<p align=\"center\"><b>Name</b></p>\n" +
			"\t</th>\n" +
			"\t<th>\n" +
			"\t<p align=\"center\"><b>Cover</b></p>\n" +
			"\t</th>\n" +
			"\t<th>\n" +
			"\t<p align=\"center\"><b>Familiar</b></p>\n" +
			"\t</th>\n" +
			"\t<th>\n" +
			"\t<p align=\"center\"><b>Expected</b></p>\n" +
			"\t</th>\n" +
			"    <th>\n" +
			"\t<p align=\"center\"><b>Liked</b></p>\n" +
			"\t</th>\n" +
			"  </tr>";

	public static String getBottom(int num, String userId) {
		return "</table><input type=\"hidden\" name=\"userId\" value=\"" + userId + "\" /><input type=\"hidden\" name=\"num\" value=\"" + num + "\" />\n" +
				"<br/>\n" +
				"<input type=\"submit\" style=\"width: 20em;  height: 2em;\">\n" +
				"</form>\n" +
				"\n" +
				"</body>\n" +
				"</html>";
	}

	public static String getLine(String[] split, int num) {
		String year = split[5];
		String id = split[0];
		String name = split[1];
		String imdbId = "http://www.imdb.com/title/tt" + split[2];
		String pic = split[4];
		String pic2 = split[20];
		String youtube = "https://www.youtube.com/results?search_query=" + name + "+" + year;
		return "<tr>\n" +
				"\t<td>\n" +
				"\t<h1>" + name + " (" + year + ")</h1>\n" +
				"\t<input type=\"hidden\" name=\"name" + num + "\" value=\"" + name + "\" />\n" +
				"\t<input type=\"hidden\" name=\"id" + num + "\" value=\"" + id + "\" />\n" +
				"\t<p><a target=\"_blank\" href=\"" + youtube + "\">Check on YouTube</a></p>\n" +
				"\t<p><a target=\"_blank\" href=\"" + imdbId + "\">Check on IMDB</a></p>\n" +
				"\t</td>\n" +
				"\t<td>\n" +
				"\t<img src=\"" + pic + "\">\n" +
				"\t<img src=\"" + pic2 + "\">\n" +
				"\t</td>\n" +
				"\t<td>\n" +
				"\t<p>I have watched <br> this movie</p>\n" +
				"\t<p><input type=\"radio\" name=\"f" + num + "\" value=\"1\">Yes<br>\n" +
				"\t<input type=\"radio\" name=\"f" + num + "\" value=\"0\">No</p>\n" +
				"\t</td>\n" +
				"\t<td>\n" +
				"\t<p>I usually watch <br> these kinds of movies</p>\n" +
				"\t<p><input type=\"radio\" name=\"e" + num + "\" value=\"1\">Yes<br>\n" +
				"\t<input type=\"radio\" name=\"e" + num + "\" value=\"0\">No</p>\n" +
				"\t</td>\n" +
				"\t<td>\n" +
				"\t<p>I like this movie or <br> I have not watched the movie, <br> but it seems interesting</p>\n" +
				"\t<p><input type=\"radio\" name=\"r" + num + "\" value=\"1\">Yes<br>\n" +
				"\t<input type=\"radio\" name=\"r" + num + "\" value=\"0\">No</p>\n" +
				"\t</td>\n" +
				"  </tr>";
	}
}
