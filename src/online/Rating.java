package online;

public class Rating {
	private String userId;
	private Movie movie;
	private Double rating;

	public Rating(String userId, Movie movie, Double rating) {
		this.userId = userId;
		this.movie = movie;
		this.rating = rating;
	}

	public String getUserId() {
		return userId;
	}

	public Movie getMovie() {
		return movie;
	}

	public Double getRating() {
		return rating;
	}


}
