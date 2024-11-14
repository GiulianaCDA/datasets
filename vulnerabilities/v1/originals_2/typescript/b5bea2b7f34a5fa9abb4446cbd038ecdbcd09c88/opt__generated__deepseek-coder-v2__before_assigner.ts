const untracker = [undefined, null];

const Assigner = function(delegate: (a: any, b: any) => any, useuntrack: boolean = true): (...args: any[]) => any {
  const assigner = (...args: any[]): any => {
    console.log({ args });
    return args.reduce((a, b) => {
      if (untracker.includes(a)) throw new TypeError(`can't convert ${a} to object`);
      if (useuntrack && untracker.includes(b)) return a;
      Object.keys(b).forEach(key => {
        if (untracker.includes(a[key])) {
          try {
            a[key] = b[key];
          } catch (e) {
            console.error(`Error assigning key ${key}:`, e);
          }
        } else {
          try {
            a[key] = delegate.call(this, a[key], b[key]);
          } catch (e) {
            console.error(`Error delegating call for key ${key}:`, e);
          }
        }
      });
      return a;
    });
  };
  return assigner;
};

Assigner.count = (qty: number, delegate: (arg: any, ...args: any[]) => any): ((...receives: any[]) => any) => {
  const assigner = (...receives: any[]): any => {
    let group = receives.shift();
    if (untracker.includes(group)) throw new TypeError(`can't convert ${group} to object`);
    
    let args = receives.splice(0, qty - 1);

    while (args.length) {
      const keys = [];
      for (const arg of args)
        for (const key of Object.keys(arg))
          if (!keys.includes(key)) keys.push(key);

      for (const key of keys)
        try {
          group[key] = delegate.call(this, group[key], ...args.map(arg => arg[key]));
        } catch (e) {
          console.error(`Error delegating call for key ${key}:`, e);
        }

      args = receives.splice(0, qty - 1);
    }

    return group;
  };
  return assigner;
};

declare namespace Assigner {
  export type BasicTypes = string | number | symbol | bigint | object | boolean | Function;
  export type Types = BasicTypes | Types[];
  export type TypesWithExclude = BasicTypes | undefined | null | TypesWithExclude[];
}

export { Assigner };
