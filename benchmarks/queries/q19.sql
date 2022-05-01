select
    p_partkey, l_partkey

from
    lineitem inner join part
on p_partkey = l_partkey
--inner join partsupp on
--p_partkey = ps_partkey
--inner join supplier on 
--ps_suppkey = s_suppkey
