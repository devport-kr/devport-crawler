"""ArticleTag model for article_tags junction table"""

from sqlalchemy import Column, BigInteger, String, ForeignKey
from app.config.database import Base


class ArticleTag(Base):
    """
    ArticleTag model for storing tags associated with articles

    This is a junction table that stores the many-to-many relationship
    between articles and their tags.
    """
    __tablename__ = "article_tags"

    article_id = Column(BigInteger, ForeignKey("articles.id", ondelete="CASCADE"), primary_key=True)
    tag = Column(String(255), primary_key=True)

    def __repr__(self):
        return f"<ArticleTag article_id={self.article_id} tag={self.tag}>"
