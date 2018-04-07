#pragma once
#include <iostream>
#include <iomanip>
#include <conio.h>


class General
{
public:
	template<typename _Tp, class _LT>
	static void sort(vector<_Tp>& vec, _LT LT = _LT())
	{
		int isort_thresh = 7;
		int sp = 0;

		struct
		{
			_Tp *lb;
			_Tp *ub;
		} stack[48];

		size_t total = vec.size();

		if(total <= 1)
			return;

		_Tp* arr = &vec[0];
		stack[0].lb = arr;
		stack[0].ub = arr + (total - 1);

		while(sp >= 0)
		{
			_Tp* left = stack[sp].lb;
			_Tp* right = stack[sp--].ub;

			for(;;)
			{
				int i, n = (int) (right - left) + 1, m;
				_Tp* ptr;
				_Tp* ptr2;

				if(n <= isort_thresh)
				{
				insert_sort:
					for(ptr = left + 1; ptr <= right; ptr++)
					{
						for(ptr2 = ptr; ptr2 > left && LT(ptr2[0], ptr2[-1]); ptr2--)
							std::swap(ptr2[0], ptr2[-1]);
					}
					break;
				} else
				{
					_Tp* left0;
					_Tp* left1;
					_Tp* right0;
					_Tp* right1;
					_Tp* pivot;
					_Tp* a;
					_Tp* b;
					_Tp* c;
					int swap_cnt = 0;

					left0 = left;
					right0 = right;
					pivot = left + (n / 2);

					if(n > 40)
					{
						int d = n / 8;
						a = left, b = left + d, c = left + 2 * d;
						left = LT(*a, *b) ? (LT(*b, *c) ? b : (LT(*a, *c) ? c : a))
							: (LT(*c, *b) ? b : (LT(*a, *c) ? a : c));

						a = pivot - d, b = pivot, c = pivot + d;
						pivot = LT(*a, *b) ? (LT(*b, *c) ? b : (LT(*a, *c) ? c : a))
							: (LT(*c, *b) ? b : (LT(*a, *c) ? a : c));

						a = right - 2 * d, b = right - d, c = right;
						right = LT(*a, *b) ? (LT(*b, *c) ? b : (LT(*a, *c) ? c : a))
							: (LT(*c, *b) ? b : (LT(*a, *c) ? a : c));
					}

					a = left, b = pivot, c = right;
					pivot = LT(*a, *b) ? (LT(*b, *c) ? b : (LT(*a, *c) ? c : a))
						: (LT(*c, *b) ? b : (LT(*a, *c) ? a : c));
					if(pivot != left0)
					{
						std::swap(*pivot, *left0);
						pivot = left0;
					}
					left = left1 = left0 + 1;
					right = right1 = right0;

					for(;;)
					{
						while(left <= right && !LT(*pivot, *left))
						{
							if(!LT(*left, *pivot))
							{
								if(left > left1)
									std::swap(*left1, *left);
								swap_cnt = 1;
								left1++;
							}
							left++;
						}

						while(left <= right && !LT(*right, *pivot))
						{
							if(!LT(*pivot, *right))
							{
								if(right < right1)
									std::swap(*right1, *right);
								swap_cnt = 1;
								right1--;
							}
							right--;
						}

						if(left > right)
							break;
						std::swap(*left, *right);
						swap_cnt = 1;
						left++;
						right--;
					}

					if(swap_cnt == 0)
					{
						left = left0, right = right0;
						goto insert_sort;
					}

					n = std::min((int) (left1 - left0), (int) (left - left1));
					for(i = 0; i < n; i++)
						std::swap(left0[i], left[i - n]);

					n = std::min((int) (right0 - right1), (int) (right1 - right));
					for(i = 0; i < n; i++)
						std::swap(left[i], right0[i - n + 1]);
					n = (int) (left - left1);
					m = (int) (right1 - right);
					if(n > 1)
					{
						if(m > 1)
						{
							if(n > m)
							{
								stack[++sp].lb = left0;
								stack[sp].ub = left0 + n - 1;
								left = right0 - m + 1, right = right0;
							} else
							{
								stack[++sp].lb = right0 - m + 1;
								stack[sp].ub = right0;
								left = left0, right = left0 + n - 1;
							}
						} else
							left = left0, right = left0 + n - 1;
					} else if(m > 1)
						left = right0 - m + 1, right = right0;
					else
						break;
				}
			}
		}
	}

	static void displayRateOfProgress(double current, double total)
	{
		static const int progressBarSize = 50;
		if((current + 1)*progressBarSize / total != current * progressBarSize / total)
		_cprintf("%.1f\%\n", current / total * 100);
		//std::cout << setiosflags(ios::fixed) << setprecision(1) << current / total * 100 << '%' << std::flush;
	}

};

