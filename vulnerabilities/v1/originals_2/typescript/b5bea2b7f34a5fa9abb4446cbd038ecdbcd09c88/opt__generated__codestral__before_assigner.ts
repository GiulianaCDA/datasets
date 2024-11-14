const untracker = [undefined, null];

type DelegateFunction = (a: Record<string, any>, b: Record<string, any>) => any;

const Assigner = function (delegate: DelegateFunction, useUntrack: boolean = true): (...args: Array<Record<string, any> | null | undefined>) => Record<string, any> {
  const assigner = (...args: Array<Record<string, any> | null | undefined>) => {
    console.log({ args });

    return args.reduce((a, b) => {
      if (!b || untracker.includes(b)) {
        return a;
      }

      Object.keys(b).forEach((key) => {
        const bValue = b[key];
        if (untracker.includes(a[key])) {
          a[key] = bValue;
        } else {
          a[key] = delegate(a, { [key]: bValue });
        }
      });

      return a;
    }, {} as Record<string, any>);
  };

  return assigner;
};

Assigner.count = (qty: number, delegate: DelegateFunction) => {
  const assigner = (...receives: Array<Record<string, any> | null | undefined>) => {
    let group = receives.shift();
    if (!group || untracker.includes(group)) {
      throw new TypeError(`Invalid group object`);
    }

    let args = receives.splice(0, qty - 1);
    while (args.length > 0) {
      const keys: string[] = [];
      for (const arg of args) {
        if (!arg || untracker.includes(arg)) {
          continue;
        }
        for (const key of Object.keys(arg)) {
          if (!keys.includes(key)) {
            keys.push(key);
          }
        }
      }

      for (const key of keys) {
        group[key] = delegate(group, ...args.map((arg) => ({ [key]: arg?.[key] ?? {} })));
      }

      args = receives.splice(0, qty - 1);
    }

    return group;
  };

  return assigner;
};
