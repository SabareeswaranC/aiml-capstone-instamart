-- Part A, task 6: seven analysis queries against data/instamart.db (SQLite 3.25+).
-- Only completed orders count as revenue or spend. Each query starts with a "-- name:" line;
-- copilot/part_a_sql.py splits the file on those lines and runs each with pd.read_sql_query.

-- name: Q1_where_then_having
-- Completed orders above INR 500 (WHERE filters rows), grouped by customer, keeping customers
-- whose total of those orders exceeds INR 8000 (HAVING filters groups).
SELECT customer_id,
       COUNT(*)                    AS orders_above_500,
       ROUND(SUM(order_amount), 2) AS total_above_500
FROM orders
WHERE status = 'completed'
  AND order_amount > 500
GROUP BY customer_id
HAVING SUM(order_amount) > 8000
ORDER BY total_above_500 DESC;

-- name: Q2_three_table_join
-- Revenue by city and product category.
SELECT c.city,
       p.category,
       ROUND(SUM(o.order_amount), 2) AS revenue
FROM orders o
JOIN customers c ON c.customer_id = o.customer_id
JOIN products  p ON p.product_id  = o.product_id
WHERE o.status = 'completed'
GROUP BY c.city, p.category
ORDER BY revenue DESC;

-- name: Q3_scalar_subquery
-- Products priced above the average catalogue price.
SELECT product_id, product_name, category, price_inr
FROM products
WHERE price_inr > (SELECT AVG(price_inr) FROM products)
ORDER BY price_inr DESC;

-- name: Q4_left_join
-- Customers who have no completed order (the status filter sits in the ON clause so that
-- customers with only cancelled orders still come back with NULLs).
SELECT c.customer_id, c.city
FROM customers c
LEFT JOIN orders o
       ON o.customer_id = c.customer_id
      AND o.status = 'completed'
WHERE o.order_id IS NULL
ORDER BY c.customer_id;

-- name: Q5_cte
-- One CTE totals each customer's completed spend, a second averages those totals;
-- the main query keeps customers above that average.
WITH customer_spend AS (
    SELECT customer_id, SUM(order_amount) AS total_spend
    FROM orders
    WHERE status = 'completed'
    GROUP BY customer_id
),
average_spend AS (
    SELECT AVG(total_spend) AS avg_spend FROM customer_spend
)
SELECT s.customer_id,
       ROUND(s.total_spend, 2) AS total_spend,
       ROUND(a.avg_spend, 2)   AS avg_spend
FROM customer_spend s
CROSS JOIN average_spend a
WHERE s.total_spend > a.avg_spend
ORDER BY s.total_spend DESC;

-- name: Q6_ranking_window
-- Total completed spend per customer, ranked inside each city three ways.
WITH spend AS (
    SELECT c.city, o.customer_id, ROUND(SUM(o.order_amount), 2) AS total_spend
    FROM orders o
    JOIN customers c ON c.customer_id = o.customer_id
    WHERE o.status = 'completed'
    GROUP BY c.city, o.customer_id
)
SELECT city, customer_id, total_spend,
       RANK()       OVER (PARTITION BY city ORDER BY total_spend DESC) AS rnk,
       DENSE_RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS dense_rnk,
       ROW_NUMBER() OVER (PARTITION BY city ORDER BY total_spend DESC) AS row_num
FROM spend
ORDER BY city, row_num;

-- name: Q7_running_total
-- Daily completed revenue with a cumulative total.
WITH daily AS (
    SELECT SUBSTR(order_ts, 1, 10) AS day, SUM(order_amount) AS revenue
    FROM orders
    WHERE status = 'completed'
    GROUP BY SUBSTR(order_ts, 1, 10)
)
SELECT day,
       ROUND(revenue, 2)                             AS revenue,
       ROUND(SUM(revenue) OVER (ORDER BY day), 2)    AS cumulative_revenue
FROM daily
ORDER BY day;
