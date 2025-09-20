# AI-Powered Movie Recommendation System

A high-performance Django-based movie recommendation system that implements advanced rating-based collaborative filtering with intelligent user similarity mapping. The system uses the MovieLens dataset and provides both web interface and API endpoints for personalized movie recommendations with detailed explanations.

## Features

- **AI-Powered Rating-Based Recommendations**: Advanced collaborative filtering that maps new users to existing users with similar rating patterns
- **High-Performance Caching**: Intelligent caching system for sub-second recommendation response times
- **Detailed Recommendation Explanations**: Each recommendation includes detailed reasoning and confidence scores
- **User Authentication**: Registration and login system with user profiles
- **Preference Management**: Users can set favorite genres and receive personalized recommendations
- **Movie Browsing**: Browse and search movies with filtering and pagination
- **Performance Evaluation**: Built-in evaluation metrics with real user data validation
- **REST API**: JSON API endpoints for programmatic access
- **Data Management**: Django management commands for loading MovieLens dataset
- **Optimized Algorithms**: Fast cosine similarity calculations with data limiting for scalability

## Technology Stack

- **Backend**: Django 5.2.6, Django REST Framework
- **Machine Learning**: scikit-learn, pandas, numpy
- **Database**: SQLite (default), supports PostgreSQL/MySQL
- **Caching**: Django Cache Framework with Redis support
- **Frontend**: Django templates with Bootstrap styling
- **Performance**: Optimized algorithms with intelligent caching

## Installation

### Prerequisites

- Python 3.8+
- pip (Python package installer)

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd minor_project
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run database migrations**
   ```bash
   python manage.py migrate
   ```

5. **Create a superuser (optional)**
   ```bash
   python manage.py createsuperuser
   ```

6. **Load MovieLens dataset**
   ```bash
   python manage.py load_movielens --path /path/to/movielens/folder
   ```
   The MovieLens folder should contain `movies.csv`, `ratings.csv`, and optionally `tags.csv`.

7. **Run the development server**
   ```bash
   python manage.py runserver
   ```

8. **Access the application**
   - Web interface: http://127.0.0.1:8000/
   - Admin panel: http://127.0.0.1:8000/admin/

## Usage

### Web Interface

1. **Registration/Login**: Create an account or login to access personalized features
2. **Dashboard**: Main hub with navigation to different features
3. **Preferences**: Set your favorite movie genres
4. **AI Recommendations**: Get intelligent movie recommendations based on your rating patterns and similar users
5. **Browse Movies**: Explore the movie database with search and filtering, rate movies to improve recommendations
6. **Accuracy Metrics**: View system performance evaluation with real user data validation

### API Endpoints

- **Get Recommendations**: `GET /api/recommendations/{user_id}/`
  ```json
  {
    "user_id": 1,
    "recommendations": [
      {
        "movie_id": 123,
        "title": "Movie Title",
        "genres": "Action|Adventure",
        "year": 2020,
        "explanation": {
          "reason": "Recommended by users with similar tastes (rating: 4.2/5.0)",
          "average_rating": 4.2,
          "rating_count": 150,
          "similarity_score": 0.75
        }
      }
    ]
  }
  ```

### Management Commands

- **Load MovieLens Data**:
  ```bash
  python manage.py load_movielens --path /path/to/movielens/folder
  ```

- **Evaluate Recommender**:
  ```bash
  python manage.py evaluate_recommender --test_ratio 0.2 --top_n 10
  ```

## Algorithm Details

### Rating-Based Collaborative Filtering (Primary System)
- **User Similarity Mapping**: Maps new users to existing MovieLens users with similar rating patterns
- **Advanced Similarity Calculation**: Uses optimized cosine similarity for fast user matching
- **Intelligent Scoring**: Combines similarity-weighted ratings with confidence scores and user preference alignment
- **Genre Boosting**: Enhances recommendations based on user's genre preferences
- **Popularity Correction**: Reduces bias toward overly popular movies

### Performance Optimizations
- **Intelligent Caching**: 1-hour cache for similarity matrix, 30-minute cache for user recommendations
- **Data Limiting**: Uses top 10K most active users instead of all 100K+ for scalability
- **Fast Algorithms**: Simplified cosine similarity instead of complex multi-method approaches
- **Database Optimization**: Optimized queries with select_related and strategic limits
- **Lazy Loading**: Only calculates what's needed, with smart fallbacks

### Recommendation Quality Features
- **Detailed Explanations**: Each recommendation includes reasoning, confidence scores, and technical details
- **Multi-Factor Scoring**: Combines rating quality, confidence, similarity, preference alignment, and popularity
- **Fallback Systems**: Graceful degradation to personalized and popular movie recommendations
- **Real-time Updates**: Cache invalidation when users add new ratings

## Project Structure

```
minor_project/
├── recommendations/           # Main Django app
│   ├── models.py             # Database models (User, Movie, Rating, Tag, UserRating)
│   ├── views.py              # Web views and API endpoints with cache invalidation
│   ├── urls.py               # URL routing
│   ├── serializers.py        # API serializers
│   ├── services/
│   │   ├── recommender.py    # Hybrid recommendation algorithms
│   │   ├── rating_recommender.py  # AI-powered rating-based recommender
│   │   ├── preference_recommender.py  # Genre-based recommendations
│   │   └── recommender.py    # Core recommendation algorithms
│   └── management/commands/  # Django management commands
├── recommender_project/      # Django project settings
├── templates/                # HTML templates with modern UI
├── static/                   # Static files (CSS, JS)
└── requirements.txt          # Python dependencies
```

## Database Models

- **UserProfile**: Extended user information with preferences and favorite genres
- **Movie**: Movie metadata (title, year, genres) with optimized indexing
- **Rating**: MovieLens dataset ratings for collaborative filtering
- **UserRating**: New user ratings for personalized recommendations
- **Tag**: User-generated tags for movies (optional)

## Configuration

The system uses Django's default settings. For production deployment:

1. Set `DEBUG = False` in `settings.py`
2. Configure a production database (PostgreSQL recommended)
3. Set up static file serving
4. Configure environment variables for sensitive settings

## Evaluation Metrics

The system provides comprehensive evaluation using real user data:
- **User Similarity**: Measures how well the system maps new users to similar existing users
- **Rating Analysis**: Evaluates recommendation quality based on user rating patterns
- **Diversity**: Measures the variety of genres in recommendations
- **Confidence Scores**: Tracks recommendation confidence and accuracy
- **Real-time Validation**: Uses actual user ratings for continuous evaluation

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is open source. Please check the license file for details.

## Dataset

This project uses the MovieLens dataset. You can download it from:
- [MovieLens Latest Small](https://grouplens.org/datasets/movielens/latest/)
- [MovieLens Latest](https://grouplens.org/datasets/movielens/latest/)

## Troubleshooting

### Common Issues

1. **Database errors**: Ensure migrations are run and database is properly configured
2. **Import errors**: Verify all dependencies are installed in the virtual environment
3. **Data loading issues**: Check that MovieLens CSV files are in the correct format
4. **Performance issues**: Consider using a more powerful database for large datasets

### Support

For issues and questions, please create an issue in the repository or contact the development team.

## Performance Characteristics

- **Cold Start Time**: < 1 second for first-time recommendations
- **Cached Response**: < 0.5 seconds for subsequent requests
- **Scalability**: Handles 100K+ ratings with optimized algorithms
- **Memory Usage**: Efficient caching reduces memory footprint
- **Database Load**: Optimized queries minimize database pressure

## Future Enhancements

- Real-time recommendation updates with WebSocket support
- Advanced machine learning models (neural networks, deep learning)
- Social features (friend recommendations, social proof)
- Mobile application with offline capabilities
- Advanced recommendation explanation with visualizations
- A/B testing framework for recommendation algorithms
- Multi-armed bandit algorithms for exploration vs exploitation
- Real-time user behavior tracking and adaptation
