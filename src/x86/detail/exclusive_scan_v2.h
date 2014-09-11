template <typename T, typename Op>
void exclusive_scan(const T *in_begin, const T *in_end, T *out_begin)
{
  if (begin == end)   return;
  T last_in = *begin++;
  *out_begin = Op.limit();
  while (in_begin < in_end) {
    T new_v = Op(*out, last_in);
    last_in = *in++;
    *out_begin++ = new_v;
  }
}

    
