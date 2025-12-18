import os
import asyncio
import json
import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferWindowMemory
from langchain.tools import BaseTool
from langchain.prompts import PromptTemplate
from sqlalchemy import select, func, text
from datetime import datetime, timedelta
from database.models import Transaction as TransactionModel, TransactionType

api_key = os.getenv("GEMINI_API_KEY")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="./chroma_db")
vectorstore = Chroma(
    client=client,
    embedding_function=embeddings,
    collection_name="user_transactions"
)

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)

# Tool for aggregating spend by category
class AggregateSpendByCategoryTool(BaseTool):
    name: str = "aggregate_spend_by_category"
    description: str = "Calculate the total, average, or count of spends for a specific category, optionally in the last X days. Provide the category, metric (sum/avg/count), and days (optional)."
    
    db: object
    user_id: int

    def _run(self, input_str: str):
        return asyncio.run(self._arun(input_str))

    async def _arun(self, input_str: str):
        try:
            params = json.loads(input_str)
            category = params.get("category")
            metric = params.get("metric", "sum")
            days = params.get("days")
            if not category or metric not in ["sum", "avg", "count"]:
                return "Error: Invalid input. 'category' required, 'metric' must be sum/avg/count."
        except json.JSONDecodeError:
            return "Error: Input must be a valid JSON string."
        
        if metric == "sum":
            agg_func = func.sum(TransactionModel.amount)
        elif metric == "avg":
            agg_func = func.avg(TransactionModel.amount)
        elif metric == "count":
            agg_func = func.count(TransactionModel.id)
        
        query = select(agg_func).where(
            TransactionModel.user_id == self.user_id,
            TransactionModel.category.ilike(category),
            TransactionModel.transaction_type == TransactionType.debit
        )
        if days:
            since = datetime.now() - timedelta(days=days)
            query = query.where(TransactionModel.transaction_date >= since)
        
        result = await self.db.execute(query)
        value = result.scalar() or 0
        period = f"last {days} days" if days else "all time"
        return f"{metric.capitalize()} spend on '{category}' in {period}: INR {value:.2f}"

# Tool for aggregating credit by category
class AggregateCreditByCategoryTool(BaseTool):
    name: str = "aggregate_credit_by_category"
    description: str = "Calculate the total, average, or count of credits for a specific category, optionally in the last X days. Provide the category, metric (sum/avg/count), and days (optional)."
    
    db: object
    user_id: int

    def _run(self, input_str: str):
        return asyncio.run(self._arun(input_str))

    async def _arun(self, input_str: str):
        try:
            params = json.loads(input_str)
            category = params.get("category")
            metric = params.get("metric", "sum")
            days = params.get("days")
            if not category or metric not in ["sum", "avg", "count"]:
                return "Error: Invalid input. 'category' required, 'metric' must be sum/avg/count."
        except json.JSONDecodeError:
            return "Error: Input must be a valid JSON string."
        
        if metric == "sum":
            agg_func = func.sum(TransactionModel.amount)
        elif metric == "avg":
            agg_func = func.avg(TransactionModel.amount)
        elif metric == "count":
            agg_func = func.count(TransactionModel.id)
        
        query = select(agg_func).where(
            TransactionModel.user_id == self.user_id,
            TransactionModel.category.ilike(category),
            TransactionModel.transaction_type == TransactionType.credit
        )
        if days:
            since = datetime.now() - timedelta(days=days)
            query = query.where(TransactionModel.transaction_date >= since)
        
        result = await self.db.execute(query)
        value = result.scalar() or 0
        period = f"last {days} days" if days else "all time"
        return f"{metric.capitalize()} credit on '{category}' in {period}: INR {value:.2f}"

# Tool for summing spends in the last X days
class SumSpendLastXDaysTool(BaseTool):
    name: str = "sum_spend_last_x_days"
    description: str = "Calculate the total spend in the last X days across all categories. Provide the number of days."
    
    db: object
    user_id: int

    def _run(self, input_str: str):
        return asyncio.run(self._arun(input_str))

    async def _arun(self, input_str: str):
        try:
            params = json.loads(input_str)
            days = params.get("days")
            if days is None or not isinstance(days, int) or days <= 0:
                return "Error: 'days' must be a positive integer."
        except json.JSONDecodeError:
            return "Error: Input must be a valid JSON string."
        
        since = datetime.now() - timedelta(days=days)
        query = select(func.sum(TransactionModel.amount)).where(
            TransactionModel.user_id == self.user_id,
            TransactionModel.transaction_type == TransactionType.debit,
            TransactionModel.transaction_date >= since
        )
        
        result = await self.db.execute(query)
        total = result.scalar() or 0
        return f"Total spend in the last {days} days: INR {total:.2f}"

# Tool for summing credits in the last X days
class SumCreditLastXDaysTool(BaseTool):
    name: str = "sum_credit_last_x_days"
    description: str = "Calculate the total credit in the last X days across all categories. Provide the number of days."
    
    db: object
    user_id: int

    def _run(self, input_str: str):
        return asyncio.run(self._arun(input_str))

    async def _arun(self, input_str: str):
        try:
            params = json.loads(input_str)
            days = params.get("days")
            if days is None or not isinstance(days, int) or days <= 0:
                return "Error: 'days' must be a positive integer."
        except json.JSONDecodeError:
            return "Error: Input must be a valid JSON string."
        
        since = datetime.now() - timedelta(days=days)
        query = select(func.sum(TransactionModel.amount)).where(
            TransactionModel.user_id == self.user_id,
            TransactionModel.transaction_type == TransactionType.credit,
            TransactionModel.transaction_date >= since
        )
        
        result = await self.db.execute(query)
        total = result.scalar() or 0
        return f"Total credit in the last {days} days: INR {total:.2f}"

# Tool for comparing spends between two periods
class ComparePeriodsTool(BaseTool):
    name: str = "compare_periods"
    description: str = "Compare total spend between two periods (e.g., months). Provide period1, period2, and optionally category."
    
    db: object
    user_id: int

    def _run(self, input_str: str):
        return asyncio.run(self._arun(input_str))

    async def _arun(self, input_str: str):
        try:
            params = json.loads(input_str)
            period1 = params.get("period1")
            period2 = params.get("period2")
            category = params.get("category")
            if not period1 or not period2:
                return "Error: 'period1' and 'period2' are required (format: YYYY-MM)."
        except json.JSONDecodeError:
            return "Error: Input must be a valid JSON string."
        
        def get_total(period):
            start = datetime.strptime(period, "%Y-%m")
            end = start + timedelta(days=31)  # Rough month end
            query = select(func.sum(TransactionModel.amount)).where(
                TransactionModel.user_id == self.user_id,
                TransactionModel.transaction_type == TransactionType.debit,
                TransactionModel.transaction_date >= start,
                TransactionModel.transaction_date < end
            )
            if category:
                query = query.where(TransactionModel.category.ilike(category))
            return query
        
        query1 = get_total(period1)
        query2 = get_total(period2)
        
        result1 = await self.db.execute(query1)
        result2 = await self.db.execute(query2)
        total1 = result1.scalar() or 0
        total2 = result2.scalar() or 0
        diff = total1 - total2
        pct = (diff / total2 * 100) if total2 else 0
        cat_str = f" for '{category}'" if category else ""
        return f"Spend in {period1}: INR {total1:.2f}{cat_str} vs. {period2}: INR {total2:.2f}{cat_str} (Difference: INR {diff:.2f}, {pct:.1f}%)"

# Tool for comparing credits between two periods
class CompareCreditPeriodsTool(BaseTool):
    name: str = "compare_credit_periods"
    description: str = "Compare total credit between two periods (e.g., months). Provide period1, period2, and optionally category."
    
    db: object
    user_id: int

    def _run(self, input_str: str):
        return asyncio.run(self._arun(input_str))

    async def _arun(self, input_str: str):
        try:
            params = json.loads(input_str)
            period1 = params.get("period1")
            period2 = params.get("period2")
            category = params.get("category")
            if not period1 or not period2:
                return "Error: 'period1' and 'period2' are required (format: YYYY-MM)."
        except json.JSONDecodeError:
            return "Error: Input must be a valid JSON string."
        
        def get_total(period):
            start = datetime.strptime(period, "%Y-%m")
            end = start + timedelta(days=31)  # Rough month end
            query = select(func.sum(TransactionModel.amount)).where(
                TransactionModel.user_id == self.user_id,
                TransactionModel.transaction_type == TransactionType.credit,
                TransactionModel.transaction_date >= start,
                TransactionModel.transaction_date < end
            )
            if category:
                query = query.where(TransactionModel.category.ilike(category))
            return query
        
        query1 = get_total(period1)
        query2 = get_total(period2)
        
        result1 = await self.db.execute(query1)
        result2 = await self.db.execute(query2)
        total1 = result1.scalar() or 0
        total2 = result2.scalar() or 0
        diff = total1 - total2
        pct = (diff / total2 * 100) if total2 else 0
        cat_str = f" for '{category}'" if category else ""
        return f"Credit in {period1}: INR {total1:.2f}{cat_str} vs. {period2}: INR {total2:.2f}{cat_str} (Difference: INR {diff:.2f}, {pct:.1f}%)"

# Tool for top categories by spend
class TopCategoriesBySpendTool(BaseTool):
    name: str = "top_categories_by_spend"
    description: str = "List top N categories by total spend, optionally in the last X days. Provide top_n and optionally days."
    
    db: object
    user_id: int

    def _run(self, input_str: str):
        return asyncio.run(self._arun(input_str))

    async def _arun(self, input_str: str):
        try:
            params = json.loads(input_str)
            top_n = params.get("top_n", 5)
            days = params.get("days")
            if not isinstance(top_n, int) or top_n <= 0:
                return "Error: 'top_n' must be a positive integer."
        except json.JSONDecodeError:
            return "Error: Input must be a valid JSON string."
        
        query = select(
            TransactionModel.category,
            func.sum(TransactionModel.amount).label("total")
        ).where(TransactionModel.user_id == self.user_id, TransactionModel.transaction_type == TransactionType.debit).group_by(TransactionModel.category).order_by(func.sum(TransactionModel.amount).desc()).limit(top_n)
        
        if days:
            since = datetime.now() - timedelta(days=days)
            query = query.where(TransactionModel.transaction_date >= since)
        
        result = await self.db.execute(query)
        rows = result.all()
        if not rows:
            return "No categories found."
        output = f"Top {len(rows)} categories by spend in {'last ' + str(days) + ' days' if days else 'all time'}: " + ", ".join([f"{row.category or 'Uncategorized'}: INR {row.total:.2f}" for row in rows])
        return output

# Tool for top categories by credit
class TopCategoriesByCreditTool(BaseTool):
    name: str = "top_categories_by_credit"
    description: str = "List top N categories by total credit, optionally in the last X days. Provide top_n and optionally days."
    
    db: object
    user_id: int

    def _run(self, input_str: str):
        return asyncio.run(self._arun(input_str))

    async def _arun(self, input_str: str):
        try:
            params = json.loads(input_str)
            top_n = params.get("top_n", 5)
            days = params.get("days")
            if not isinstance(top_n, int) or top_n <= 0:
                return "Error: 'top_n' must be a positive integer."
        except json.JSONDecodeError:
            return "Error: Input must be a valid JSON string."
        
        query = select(
            TransactionModel.category,
            func.sum(TransactionModel.amount).label("total")
        ).where(TransactionModel.user_id == self.user_id, TransactionModel.transaction_type == TransactionType.credit).group_by(TransactionModel.category).order_by(func.sum(TransactionModel.amount).desc()).limit(top_n)
        
        if days:
            since = datetime.now() - timedelta(days=days)
            query = query.where(TransactionModel.transaction_date >= since)
        
        result = await self.db.execute(query)
        rows = result.all()
        if not rows:
            return "No categories found."
        output = f"Top {len(rows)} categories by credit in {'last ' + str(days) + ' days' if days else 'all time'}: " + ", ".join([f"{row.category or 'Uncategorized'}: INR {row.total:.2f}" for row in rows])
        return output

# Tool for listing available categories
class ListCategoriesTool(BaseTool):
    name: str = "list_categories"
    description: str = "List all distinct categories that have transactions for the user. Input: No input required (just an empty string)."
    
    db: object
    user_id: int

    def _run(self, input_str: str):
        return asyncio.run(self._arun(input_str))

    async def _arun(self, input_str: str):
        query = select(TransactionModel.category).where(TransactionModel.user_id == self.user_id).distinct()
        result = await self.db.execute(query)
        categories = [row.category for row in result if row.category]
        if not categories:
            return "No categories found."
        return f"Available categories: {', '.join(categories)}"

# Tool for executing custom SELECT SQL queries
class SQLQueryTool(BaseTool):
    name: str = "sql_query"
    description: str = "Execute a custom SELECT SQL query on the transactions table. The table has columns: id (int, primary key), user_id (int), amount (decimal), transaction_type (enum: 'credit' or 'debit'), transaction_date (datetime), description (string, optional), category (string, optional). The query will be automatically scoped to the current user. Input: A SELECT SQL string (e.g., 'SELECT SUM(amount) FROM transactions WHERE category = \"food\"'). Only SELECT queries are allowed."
    
    db: object
    user_id: int

    def _run(self, sql: str):
        return asyncio.run(self._arun(sql))

    async def _arun(self, sql: str):
        if not sql.strip().upper().startswith("SELECT"):
            return "Error: Only SELECT queries are allowed."
        
        # Insert user filter properly
        if "WHERE" in sql.upper():
            sql = sql.replace("WHERE", f"WHERE user_id = {self.user_id} AND ", 1)
        else:
            sql = sql.replace("FROM transactions", f"FROM transactions WHERE user_id = {self.user_id}", 1)
        
        try:
            result = await self.db.execute(text(sql))
            rows = result.all()
            if not rows:
                return "No results."
            # Format as simple text
            output = "\n".join([str(row._asdict()) for row in rows])
            return f"Query results:\n{output}"
        except Exception as e:
            return f"Error executing query: {str(e)}"

# Global memory store for conversation history per user
memory_store = {}

# Custom prompt for the agent
prompt_template = """
You are a financial advisor AI. Use the available tools to analyze the user's transaction data and answer their query accurately.
If the query involves specific categories, first use the list_categories tool to see what categories exist for the user.
If calculations are needed, call the appropriate tools. For complex queries, use the sql_query tool to write and execute SELECT statements.
Provide concise, data-driven advice based on the results.
If no tools are needed, answer directly.

User Query: {input}

{agent_scratchpad}
"""

# Create agent with tools
async def query_rag(query: str, db, user_id: int):
    tools = [
        AggregateSpendByCategoryTool(db=db, user_id=user_id),
        SumSpendLastXDaysTool(db=db, user_id=user_id),
        ComparePeriodsTool(db=db, user_id=user_id),
        TopCategoriesBySpendTool(db=db, user_id=user_id),
        ListCategoriesTool(db=db, user_id=user_id),
        SQLQueryTool(db=db, user_id=user_id),
        AggregateCreditByCategoryTool(db=db, user_id=user_id),
        SumCreditLastXDaysTool(db=db, user_id=user_id),
        CompareCreditPeriodsTool(db=db, user_id=user_id),
        TopCategoriesByCreditTool(db=db, user_id=user_id),
    ]
    
    # Get or create memory for the user
    if user_id not in memory_store:
        memory_store[user_id] = ConversationBufferWindowMemory(k=10)  # Keep last 10 messages
    memory = memory_store[user_id]
    
    # Prepend conversation history to the query
    history = memory.buffer_as_str
    if history:
        query = f"{history}\n\nCurrent query: {query}"
    
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        memory=memory,
        handle_parsing_errors=True,
        prompt=prompt_template,
        verbose=True  # Set to False for production
    )
    
    return await agent.arun(query)

# Keep the old functions for adding to RAG (unchanged)
def add_transaction_to_rag(transaction):
    """
    Add a transaction to the vector DB for retrieval.
    - Embeds the description and category as searchable text.
    - Stores metadata for context in responses.
    """
    text = f"{transaction.description or ''} {transaction.category or ''}".strip()
    if not text:
        return  # Skip if no searchable text

    meta = {
        "id": transaction.id,
        "amount": str(transaction.amount),
        "transaction_type": transaction.transaction_type.value,
        "transaction_date": transaction.transaction_date.isoformat(),
        "description": transaction.description or "",
        "category": transaction.category or ""
    }

    vectorstore.add_texts([text], metadatas=[meta], ids=[str(transaction.id)])

